from typing import Optional, Tuple

import lightning as L
import torch
from torch import Tensor
from torch_geometric.utils import scatter

from src.models.visnet_ny.kerne import ViSNetBlock, Distance

class Hoved(L.LightningModule):
    def __init__(self,
                 hidden_channels: int = 128,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 out_channels = 19,
                 ):
        super().__init__()
        self.motor = torch.nn.Linear(in_features=hidden_channels,
                                     out_features=out_channels)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op

    def forward(self, z, pos, batch, x, v):
        x = self.motor(x)
        x = x * self.std


        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean
        return y

class VisNetDownstream(L.LightningModule):

    def __init__(self,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 out_channels=19,
                 max_z: int = 100,
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
                 ):
        super().__init__()
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
            reduce_op=reduce_op,
            mean=mean,
            std=std,
            out_channels=out_channels,
            hidden_channels=hidden_channels
        )

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        x, v, edge_attr = self.rygrad(z, pos, batch,
                                      edge_index, edge_weight, edge_vec)
        y = self.hoved(z, pos, batch, x, v)
        return y