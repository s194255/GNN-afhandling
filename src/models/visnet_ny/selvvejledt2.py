from typing import Optional, Tuple

import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter

from src.models.visnet_ny.kerne import EquivariantScalar, Atomref, ViSNetBlock, Distance
from src.models.grund import GrundSelvvejledt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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
        self.motor = torch.nn.Linear(in_features=hidden_channels, out_features=3)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op
        self.derivative = derivative

    def reset_parameters(self):
        self.motor.reset_parameters()
        self.prior_model.reset_parameters()

    def forward(self, z, pos, batch, x, v):
        x = self.motor(x)
        x = x * self.std
        x = x + self.mean
        return x


class VisNetSelvvejledt2(GrundSelvvejledt):
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
            derivative=self.derivative
        )

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # TODO: få masker ind
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        x, v, edge_attr = self.rygrad(z, pos, batch,
                                      edge_index, edge_weight, edge_vec)
        x = self.hoved(z, pos, batch, x, v)
        target = 100*torch.ones(x.shape, device=self.device)
        return x, target

