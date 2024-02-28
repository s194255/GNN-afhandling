from typing import Optional, Tuple

import lightning as L
import torch
from torch import Tensor
from torch_geometric.utils import scatter

from src.models.visnet_ny.kerne import ViSNet

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

class VisNetDownstream(ViSNet):

    def __init__(self, *args,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 out_channels=19,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hoved = Hoved(
            reduce_op=reduce_op,
            mean=mean,
            std=std,
            out_channels=out_channels,
            **kwargs
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