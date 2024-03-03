import random
from typing import Optional, Tuple, Any

import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter

from src.models.visnet_ny.kerne import EquivariantScalar, Atomref, ViSNetBlock, Distance
from src.models.visnet_ny.riemannGaussian import RiemannGaussian
from src.models.grund import GrundSelvvejledt
import numpy as np

class Global(L.LightningModule):
    def __init__(self,
                 hidden_channels: int = 128,
                 out_channels: int = 4,
                 ):
        super().__init__()
        self.motor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, no) -> torch.Tensor:
        x = self.motor(x)

        return x



class Lokal(L.LightningModule):

    def __init__(self,
                 hidden_channels: int = 128,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 derivative: bool = False,
                 ):
        super().__init__()
        self.motor = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op
        self.derivative = derivative

    def reset_parameters(self):
        self.motor.reset_parameters()
        self.prior_model.reset_parameters()

    def forward(self, z, pos, batch, x, v):
        x = self.motor(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

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

class Hoved(L.LightningModule):

    def __init__(self,
                 hidden_channels: int = 128,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 derivative: bool = False,
                 ):
        super().__init__()
        self.lokal = Lokal(
            hidden_channels=hidden_channels,
            atomref=atomref,
            max_z=max_z,
            reduce_op=reduce_op,
            mean=mean,
            std=std,
            derivative=derivative,
        )
        self.globall = Global(
            hidden_channels=hidden_channels
        )

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale):
        noise_idx = random.choice(np.arange(len(self.noise_scales)))
        # lokal_udgangsdata = self.lokal(z, pos, batch, x, v)
        global_udgangsdata = self.globall(x, noise_idx)



class VisNetSelvvejledt(GrundSelvvejledt):
    def __init__(self, *args,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 lmax: int = 1,
                 vecnorm_type: Optional[str] = None,
                 trainable_vecnorm: bool = False,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 hidden_channels: int = 128,
                 num_rbf: int = 32,
                 trainable_rbf: bool = False,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 vertex: bool = False,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.derivative = True
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.rygrad = ViSNetBlock(
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

        self.hoved = Hoved(
            atomref=atomref,
            max_z=max_z,
            reduce_op=reduce_op,
            mean=mean,
            std=std,
            derivative=self.derivative,
            hidden_channels=hidden_channels
        )

        self.riemannGaussian = RiemannGaussian()
        self.noise_scales = [0.01, 0.1, 1.0, 1.0]



    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # TODO: få masker ind
        noise_idx = random.choice(np.arange(len(self.noise_scales)))
        noise_scale = self.noise_scales[noise_idx]
        pos_til, target = self.riemannGaussian(pos, noise_scale=noise_scale)
        if self.derivative:
            pos_til.requires_grad_(True)
        edge_index, edge_weight, edge_vec = self.distance(pos_til, batch)
        x, v, edge_attr = self.rygrad(z, pos_til, batch,
                                      edge_index, edge_weight, edge_vec)
        y, dy = self.hoved(z, pos_til, batch, x, v, noise_idx, noise_scale)
        # target = 100*torch.ones(y.shape, device=self.device)
        loss = noise_scale*self.criterion(1/noise_scale*dy, target)

        return loss

