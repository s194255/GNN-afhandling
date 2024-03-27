from typing import Optional

import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter

from src.models.visnet import EquivariantScalar, Atomref


class HovedSelvvejledt(L.LightningModule):
    args = {
        'atomref': None,
        'max_z': 100,
        'reduce_op': "sum",
        'mean': 0.0,
        'std': 1.0,
    }
    def __init__(self,
                 out_channels,
                 hidden_channels: int,
                 atomref: Optional[Tensor] = args['atomref'],
                 max_z: int = args['max_z'],
                 reduce_op: str = args['reduce_op'],
                 mean: float = args['mean'],
                 std: float = args['std'],
                 ):
        super().__init__()
        self.lokal = LokaltGradient(
            hidden_channels=hidden_channels,
            atomref=atomref,
            max_z=max_z,
            reduce_op=reduce_op,
            mean=mean,
            std=std,
        )
        # self.lokal = LokaltLineær(
        #     hidden_channels=hidden_channels,
        #     atomref=atomref,
        #     max_z=max_z,
        #     reduce_op=reduce_op,
        #     mean=mean,
        #     std=std,
        # )
        self.globall = Globalt(
            hidden_channels=hidden_channels,
            reduce_op=reduce_op,
            out_channels=out_channels
        )
        self.derivative = True

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        tabsopslag = {}
        tabsopslag['lokalt'] = self.lokal(z, pos, batch, x, v, noise_idx, noise_scale, target)
        tabsopslag['globalt'] = self.globall(z, pos, batch, x, v, noise_idx, noise_scale)
        return tabsopslag


class LokaltGradient(L.LightningModule):

    def __init__(self,
                 hidden_channels: int = 128,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 ):
        super().__init__()
        self.motor = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op
        self.criterion = torch.nn.MSELoss(reduction='none')

    def reset_parameters(self):
        self.motor.reset_parameters()
        self.prior_model.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        x = self.motor(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

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
        noise_scale = torch.gather(noise_scale, 0, batch)
        loss = noise_scale**2 * self.criterion(1 / noise_scale.view(-1, 1) * dy, target).sum(dim=1)
        loss = loss.mean()
        return loss


class LokaltLineær(L.LightningModule):

    def __init__(self,
                 hidden_channels: int = 128,
                 atomref: Optional[Tensor] = None,
                 max_z: int = 100,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 ):
        super().__init__()
        self.motor = torch.nn.Linear(in_features=hidden_channels, out_features=3)
        # self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.reduce_op = reduce_op
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


class Globalt(L.LightningModule):
    def __init__(self,
                 hidden_channels: int = 128,
                 out_channels: int = 4,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        self.reduce_op = reduce_op
        self.motor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale) -> torch.Tensor:
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)
        loss = self.criterion(x, noise_idx)
        return loss



