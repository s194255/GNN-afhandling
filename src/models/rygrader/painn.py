import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

import lightning as L
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from torch.nn import functional as F
from torch_geometric.data import Data
from src.models.rygrader.visnet import Distance

# link = https://gist.github.com/mjhong0708/9187130f67896de38273875c2cacbb43

Tensor = torch.Tensor
AtomsGraphDict = Dict[str, Tensor]
_default_dtype = torch.get_default_dtype()


# ===================================
#             Utilities
# ===================================


def atoms_to_data(atoms: Atoms, cutoff: float = 5.0) -> Data:
    """Convert an ASE Atoms object to a torch_geometric Data object.
    Build neighbor list.
    """
    elems = torch.tensor(atoms.numbers, dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=_default_dtype)
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError("AtomsGraph does not support partial pbc")
    pbc = atoms.pbc.all()
    if pbc:
        cell = torch.tensor(atoms.cell.array, dtype=_default_dtype).unsqueeze(0)
    else:
        cell = torch.zeros((1, 3, 3), dtype=_default_dtype)
    n_atoms = torch.tensor(len(atoms), dtype=torch.long)

    idx_i, idx_j, shift = neighbor_list("ijS", atoms, cutoff)
    idx_i = torch.tensor(idx_i, dtype=torch.long)
    idx_j = torch.tensor(idx_j, dtype=torch.long)
    shift = torch.tensor(shift, dtype=_default_dtype)
    edge_index = torch.stack([idx_j, idx_i], dim=0)
    edge_shift = shift
    batch = torch.zeros_like(elems, dtype=torch.long)

    kwargs = {}
    try:
        kwargs["energy"] = torch.tensor(atoms.get_potential_energy(), dtype=_default_dtype)
    except RuntimeError:
        pass
    try:
        kwargs["force"] = torch.tensor(atoms.get_forces(), dtype=_default_dtype)
    except RuntimeError:
        pass

    data = Data(
        elems=elems,
        pos=pos,
        cell=cell,
        n_atoms=n_atoms,
        edge_index=edge_index,
        edge_shift=edge_shift,
        batch=batch,
        **kwargs,
    )
    return data


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "silu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        activate_final: bool = False,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.activation = torch.nn.SiLU if activation == "silu" else None
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers
        layers = []
        layers.append(torch.nn.Linear(n_input, hidden_layers[0]))
        layers.append(self.activation(**activation_kwargs))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation(**activation_kwargs))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))
        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))

        self.net = torch.nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ===================================
#        Activation functions
# ===================================
class ShiftedSoftplus(torch.nn.Module):
    r"""Shifted version of softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


# ===================================
#         Cutoff functions
# ===================================


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=_default_dtype))

    def forward(self, x: Tensor) -> Tensor:
        out = 0.5 * (1 + torch.cos(torch.pi * x / self.cutoff))
        mask = x <= self.cutoff
        return out * mask


# ===================================
#                RBF
# ===================================
class BesselRBF(torch.nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf
        freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs: Tensor) -> Tensor:
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y


# ===================================
#              Layers
# ===================================
class PaiNNInteraction(torch.nn.Module):
    r"""Copied from schnetpack.
    PaiNN interaction block for modeling equivariant interactions"""

    def __init__(self, n_atom_basis: int, activation: str):

        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.interatomic_context_net = MLP(
            n_input=n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )

    def forward(
        self, q: Tensor, mu: Tensor, Wij: Tensor, dir_ij: Tensor, idx_i: Tensor, idx_j: Tensor, n_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute interaction output."""
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms, dim=0)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms, dim=0)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(torch.nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: str, epsilon: float = 1e-8):
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.intraatomic_context_net = MLP(
            n_input=2 * n_atom_basis,
            n_output=3 * n_atom_basis,
            hidden_layers=(n_atom_basis,),
            activation=activation,
        )
        self.mu_channel_mix = torch.nn.Linear(n_atom_basis, 2 * n_atom_basis, bias=False)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mu_channel_mix.weight.data)

    def forward(self, q: Tensor, mu: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute intraatomic mixing."""
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNNRepresentation(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
    ):

        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)

        self.radial_basis = BesselRBF(n_rbf, self.cutoff)
        self.embedding = torch.nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.filter_net = torch.nn.Linear(
            self.radial_basis.n_rbf,
            self.n_interactions * n_atom_basis * 3,
        )

        self.interactions = torch.nn.ModuleList(
            [PaiNNInteraction(n_atom_basis=self.n_atom_basis, activation=activation) for _ in range(self.n_interactions)]
        )
        self.mixing = torch.nn.ModuleList(
            [
                PaiNNMixing(n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon)
                for _ in range(self.n_interactions)
            ]
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.filter_net.weight.data)

    def forward(self, atoms_graph: AtomsGraphDict) -> AtomsGraphDict:
        """Compute atomic representations/embeddings."""
        # get tensors from input dictionary
        z = atoms_graph["elems"]
        edge_index = atoms_graph["edge_index"]  # neighbors
        edge_vec = atoms_graph["edge_vec"]
        idx_i = edge_index[1]
        idx_j = edge_index[0]

        n_atoms = z.size(0)

        # compute atom and pair features
        d_ij = torch.norm(edge_vec, dim=1, keepdim=True)
        dir_ij = edge_vec / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing, strict=True)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        atoms_graph["node_features"] = q
        atoms_graph["node_vec_features"] = mu
        return atoms_graph


class PaiNN(L.LightningModule):
    def __init__(
        self,
        n_atom_basis: int = 128,  # number of basis functions for node features
        n_interactions: int = 3,  # number of interaction layers
        n_rbf: int = 20,  # number of radial basis functions for edge features
        cutoff=5.0,  # cutoff radius for neighbor list
        activation: str = "silu",  # activation function
        epsilon: float = 1e-8,  # epsilon for numerical stability
        energy_mean: float = 0.0,  # mean of training set energy
        energy_std: float = 1.0,  # std dev of training set energy
        energy_loss_weight: float = 1.0,  # weight for energy loss
        force_loss_weight: float = 100.0,  # weight for force loss
        per_atom_energy_loss: bool = False,  # compute energy loss per atom
        learning_rate: float = 1e-3,  # learning rate
        reduce_lr_patience: int = 10,  # patience for learning rate scheduler
        reduce_lr_factor: float = 0.5,  # factor for learning rate scheduler
    ):
        super().__init__()
        self.cutoff = cutoff
        self.representation = PaiNNRepresentation(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            epsilon=epsilon,
        )
        self.decoder = MLP(n_input=n_atom_basis, n_output=1, hidden_layers=(n_atom_basis // 2, n_atom_basis // 2))
        self.register_buffer("energy_mean", torch.tensor(energy_mean))
        self.register_buffer("energy_std", torch.tensor(energy_std))
        self.energy_loss_weight = energy_loss_weight
        self.force_loss_weight = force_loss_weight
        self.per_atom_energy_loss = per_atom_energy_loss
        self.learning_rate = learning_rate
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.save_hyperparameters()

    def forward(self, atoms_graph: AtomsGraphDict) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        # Attributes
        atoms_graph["pos"].requires_grad_(True)
        idx_j, idx_i = atoms_graph["edge_index"][0], atoms_graph["edge_index"][1]
        edge_batch = atoms_graph["batch"][idx_i]
        atoms_graph["edge_vec"] = atoms_graph["pos"][idx_j] - atoms_graph["pos"][idx_i]
        atoms_graph["edge_vec"] += torch.einsum("ni,nij->nj", atoms_graph["edge_shift"], atoms_graph["cell"][edge_batch])
        batch_size = atoms_graph["batch"].max().item() + 1

        # Representation
        atoms_graph = self.representation(atoms_graph)

        # Energy
        E_i = self.decoder(atoms_graph["node_features"])
        E = scatter_add(E_i, atoms_graph["batch"], dim=0, dim_size=batch_size).squeeze()
        E = (E * self.energy_std) + self.energy_mean

        # Force
        fwd_outputs: List[Tensor] = [E]
        inputs: List[Tensor] = [atoms_graph["pos"]]
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(E)]
        engrad = torch.autograd.grad(fwd_outputs, inputs, grad_outputs=grad_outputs, create_graph=True)[0]
        if engrad is None:
            engrad = torch.zeros_like(atoms_graph["pos"])
        F = -engrad

        output["energy"] = E
        output["force"] = F
        return output

    def compute_loss(self, batch, output):
        if self.per_atom_energy_loss:
            n_atoms = batch["n_atoms"]
            energy_mse = torch.nn.functional.mse_loss(output["energy"] / n_atoms, batch["energy"] / n_atoms)
        else:
            energy_mse = torch.nn.functional.mse_loss(output["energy"].sum(), batch["energy"].sum())
        force_loss = torch.nn.functional.mse_loss(output["force"].ravel(), batch["force"].ravel())
        loss = self.energy_loss_weight * energy_mse + self.force_loss_weight * force_loss
        return loss

    def compute_metrics(self, batch, output):
        energy_true = batch["energy"]
        energy_pred = output["energy"]
        per_atom_energy_true = batch["energy"] / batch["n_atoms"]
        per_atom_energy_pred = output["energy"] / batch["n_atoms"]
        force_true = batch["force"].ravel()
        force_pred = output["force"].ravel()

        energy_mae = torch.nn.functional.l1_loss(energy_pred, energy_true)
        energy_rmse = torch.sqrt(torch.nn.functional.mse_loss(energy_pred, energy_true))
        per_atom_energy_mae = torch.nn.functional.l1_loss(per_atom_energy_pred, per_atom_energy_true)
        per_atom_energy_rmse = torch.sqrt(torch.nn.functional.mse_loss(per_atom_energy_pred, per_atom_energy_true))
        force_mae = torch.nn.functional.l1_loss(force_pred, force_true)
        force_rmse = torch.sqrt(torch.nn.functional.mse_loss(output["force"].ravel(), batch["force"].ravel()))
        return {
            "energy_mae": energy_mae,
            "energy_rmse": energy_rmse,
            "per_atom_energy_mae": per_atom_energy_mae,
            "per_atom_energy_rmse": per_atom_energy_rmse,
            "force_mae": force_mae,
            "force_rmse": force_rmse,
        }

    def training_step(self, batch, batch_idx):
        batch_size = batch["batch"].max().item() + 1
        with torch.inference_mode(False):
            output = self(batch)
            loss = self.compute_loss(batch, output)
            metrics = self.compute_metrics(batch, output)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
            self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
            self.log("train_energy_mae", metrics["energy_mae"], prog_bar=True, batch_size=batch_size)
            self.log("train_force_mae", metrics["force_mae"], prog_bar=True, batch_size=batch_size)
            self.log("train_energy_rmse", metrics["energy_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("train_force_rmse", metrics["force_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("train_per_atom_energy_mae", metrics["per_atom_energy_mae"], prog_bar=False, batch_size=batch_size)
            self.log("train_per_atom_energy_rmse", metrics["per_atom_energy_rmse"], prog_bar=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["batch"].max().item() + 1
        with torch.inference_mode(False):
            output = self(batch)
            loss = self.compute_loss(batch, output)
            metrics = self.compute_metrics(batch, output)
            self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
            self.log("val_energy_mae", metrics["energy_mae"], prog_bar=True, batch_size=batch_size)
            self.log("val_force_mae", metrics["force_mae"], prog_bar=True, batch_size=batch_size)
            self.log("val_energy_rmse", metrics["energy_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("val_force_rmse", metrics["force_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("val_per_atom_energy_mae", metrics["per_atom_energy_mae"], prog_bar=False, batch_size=batch_size)
            self.log("val_per_atom_energy_rmse", metrics["per_atom_energy_rmse"], prog_bar=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch["batch"].max().item() + 1
        with torch.inference_mode(False):
            output = self(batch)
            loss = self.compute_loss(batch, output)
            metrics = self.compute_metrics(batch, output)
            self.log("test_loss", loss, prog_bar=True, batch_size=batch_size)
            self.log("test_energy_mae", metrics["energy_mae"], prog_bar=True, batch_size=batch_size)
            self.log("test_force_mae", metrics["force_mae"], prog_bar=True, batch_size=batch_size)
            self.log("test_energy_rmse", metrics["energy_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("test_force_rmse", metrics["force_rmse"], prog_bar=False, batch_size=batch_size)
            self.log("test_per_atom_energy_mae", metrics["per_atom_energy_mae"], prog_bar=False, batch_size=batch_size)
            self.log("test_per_atom_energy_rmse", metrics["per_atom_energy_rmse"], prog_bar=False, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.reduce_lr_patience, factor=self.reduce_lr_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "frequency": 1},
        }


class PaiNNCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, model: PaiNN, device="cpu", **kwargs):
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.model = model.to(device)
        self.r_cut = self.model.cutoff

    def calculate(self, atoms: Atoms = None, properties=["energy"], system_changes=all_changes):  # noqa
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        atoms_graph: Dict[str, Tensor] = atoms_to_data(atoms, self.r_cut).to(self.device).to_dict()
        output = self.model(atoms_graph)
        self.results["energy"] = output["energy"].item()
        self.results["forces"] = output["force"].detach().cpu().numpy()


class PaiNNRygrad(torch.nn.Module):
    """PaiNN - polarizable interaction neural network"""

    def __init__(
        self,
        hidden_channels: int = 128,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        epsilon: float = 1e-8,
        max_num_neighbors: int = 32,
        max_z: int = 100,
    ):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)

        self.radial_basis = BesselRBF(n_rbf, self.cutoff)
        self.embedding = torch.nn.Embedding(max_z, hidden_channels, padding_idx=0)

        self.filter_net = torch.nn.Linear(
            self.radial_basis.n_rbf,
            self.n_interactions * hidden_channels * 3,
        )

        self.interactions = torch.nn.ModuleList(
            [PaiNNInteraction(n_atom_basis=self.hidden_channels, activation=activation) for _ in range(self.n_interactions)]
        )
        self.mixing = torch.nn.ModuleList(
            [
                PaiNNMixing(n_atom_basis=self.hidden_channels, activation=activation, epsilon=epsilon)
                for _ in range(self.n_interactions)
            ]
        )
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.filter_net.weight.data)

    def forward(self, z, pos, batch, edge_index):
        """Compute atomic representations/embeddings."""
        # get tensors from input dictionary
        # edge_index, edge_weight, edge_vec = self.distance(pos, batch)

        # atoms_graph["pos"].requires_grad_(True)
        idx_j, idx_i = edge_index[0], edge_index[1]
        edge_batch = batch[idx_i]
        edge_vec = pos[idx_j] - pos[idx_i]
        # edge_vec += torch.einsum("ni,nij->nj", data.edge_shift,
        #                                         data.cell[edge_batch])

        # edge_vec = data.edge_vec



        idx_i = edge_index[1]
        idx_j = edge_index[0]

        n_atoms = z.size(0)

        # compute atom and pair features
        d_ij = torch.norm(edge_vec, dim=1, keepdim=True)
        dir_ij = edge_vec / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        filter_list = torch.split(filters, 3 * self.hidden_channels, dim=-1)

        q = self.embedding(z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing, strict=True)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        x = q
        v = mu
        edge_attr = None
        masker = None
        return x, v, edge_attr, masker
