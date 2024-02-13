from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter, subgraph

from src.models.tg_redskaber import ViSNetBlock, EquivariantScalar, Atomref, Distance

class VisNetBase(torch.nn.Module):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 19,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[Tensor] = None,
        reduce_op: str = "sum",
        mean: float = 0.0,
        std: float = 1.0,
        derivative: bool = False,
    ) -> None:
        super().__init__()

        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        self.output_model = EquivariantScalar(hidden_channels=hidden_channels, out_channels=out_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.reduce_op = reduce_op
        self.derivative = derivative

        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            x (torch.Tensor): Predicted node representations
        """
        if self.derivative:
            pos.requires_grad_(True)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        x, v, edge_attr = self.representation_model(z, pos, batch,
                                                    edge_index, edge_weight, edge_vec)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        return x, edge_attr, edge_index


class ViSNet(VisNetBase):
    r"""A :pytorch:`PyTorch` module that implements the equivariant
    vector-scalar interactive graph neural network (ViSNet) from the
    `"Enhancing Geometric Representations for Molecules with Equivariant
    Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/pdf/2210.16518.pdf>`_ paper.

    Args:
        lmax (int, optional): The maximum degree of the spherical harmonics.
            (default: :obj:`1`)
        vecnorm_type (str, optional): The type of normalization to apply to the
            vectors. (default: :obj:`None`)
        trainable_vecnorm (bool, optional):  Whether the normalization weights
            are trainable. (default: :obj:`False`)
        num_heads (int, optional): The number of attention heads.
            (default: :obj:`8`)
        num_layers (int, optional): The number of layers in the network.
            (default: :obj:`6`)
        hidden_channels (int, optional): The number of hidden channels in the
            node embeddings. (default: :obj:`128`)
        num_rbf (int, optional): The number of radial basis functions.
            (default: :obj:`32`)
        trainable_rbf (bool, optional): Whether the radial basis function
            parameters are trainable. (default: :obj:`False`)
        max_z (int, optional): The maximum atomic numbers.
            (default: :obj:`100`)
        cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors
            considered for each atom. (default: :obj:`32`)
        vertex (bool, optional): Whether to use vertex geometric features.
            (default: :obj:`False`)
        atomref (torch.Tensor, optional): A tensor of atom reference values,
            or :obj:`None` if not provided. (default: :obj:`None`)
        reduce_op (str, optional): The type of reduction operation to apply
            (:obj:`"sum"`, :obj:`"mean"`). (default: :obj:`"sum"`)
        mean (float, optional): The mean of the output distribution.
            (default: :obj:`0.0`)
        std (float, optional): The standard deviation of the output
            distribution. (default: :obj:`1.0`)
        derivative (bool, optional): Whether to compute the derivative of the
            output with respect to the positions. (default: :obj:`False`)
    """

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            y (torch.Tensor): The energies or properties for each molecule.
            dy (torch.Tensor, optional): The negative derivative of energies.
        """
        x, _, _ = super().forward(z, pos, batch)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

        if self.derivative:
            grad_outputs = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction.")
            return y, -dy

        return y, None

class VisNetSelvvejledt(VisNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_channels = kwargs.get('hidden_channels', 128)
        self.edge_out = torch.nn.Linear(hidden_channels, 1)
        self.maskeringsandel = 0.15
        self.maskemager = Maskemager()
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.derivative:
            pos.requires_grad_(True)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        masker = self.maskemager(z.shape[0], edge_index, self.get_maskeringsandel())
        x, v, edge_attr = self.representation_model(z, pos, batch,
                                                    edge_index, edge_weight, edge_vec, masker)

        edge_attr = self.edge_out(edge_attr)
        edge_attr = edge_attr[masker['kanter']]
        edge_index = edge_index[:, masker['kanter']]
        y = pos[edge_index[0, :], :] - pos[edge_index[1, :], :]
        y = y.square().sum(dim=1, keepdim=True)
        return edge_attr, y

    def get_maskeringsandel(self) -> float:
        return self.maskeringsandel


class Maskemager(torch.nn.Module):

    def forward(self, n_knuder: int,
                edge_index: Tensor,
                maskeringsandel: float) -> Tuple[Tensor, Tensor]:
        randperm = torch.randperm(n_knuder)
        k = int(maskeringsandel*n_knuder)
        udvalgte_knuder = randperm[:k]
        edge_index2, _, kantmaske = subgraph(udvalgte_knuder, edge_index, return_edge_mask=True)
        idxs = torch.arange(n_knuder)
        knudemaske = torch.isin(idxs, edge_index2)
        # x[knudemaske] = torch.zeros(x.shape[1])
        # edge_attr[kantmaske] = torch.zeros(edge_attr.shape[1])
        return {'knuder': knudemaske, 'kanter': kantmaske}




