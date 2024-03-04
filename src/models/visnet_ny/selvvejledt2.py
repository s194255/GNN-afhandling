from typing import Optional

import lightning as L
import torch
from torch import Tensor

from src.models.visnet_ny.kerne import Atomref, ViSNetBlock, Distance
from src.models.grund import GrundSelvvejledt
from src.models.visnet_ny.selvvejldt import Global

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
        self.motor = torch.nn.Linear(in_features=hidden_channels, out_features=3)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op
        self.derivative = derivative
        self.criterion = torch.nn.MSELoss(reduction='none')

    def reset_parameters(self):
        self.motor.reset_parameters()
        self.prior_model.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        x = self.motor(x)
        x = x * self.std
        x = x + self.mean
        noise_scale = torch.gather(noise_scale, 0, batch)
        loss = noise_scale ** 2 * self.criterion(1 / noise_scale.view(-1, 1) * x, target).sum(dim=1)
        loss = loss.mean()
        return loss

class Hoved(L.LightningModule):

    def __init__(self,
                 hidden_channels: int = 128,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 derivative: bool = False,
                 out_channels: int = 4,
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
            hidden_channels=hidden_channels,
            reduce_op=reduce_op,
            out_channels=out_channels
        )

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        tabsopslag = {}
        tabsopslag['lokalt'] = self.lokal(z, pos, batch, x, v, noise_idx, noise_scale, target)
        tabsopslag['globalt'] = self.globall(z, pos, batch, x, v, noise_idx, noise_scale)
        return tabsopslag


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
            hidden_channels=hidden_channels,
            out_channels=len(self.noise_scales_options)
        )

