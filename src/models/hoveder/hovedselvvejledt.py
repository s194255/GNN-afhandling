from typing import Optional

import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter

from src.models.rygrader.visnet import EquivariantScalar, Atomref


class HovedSelvvejledt(L.LightningModule):
    def __init__(self,
                 out_channels,
                 hidden_channels: int,
                 lokalt: dict,
                 globalt: dict,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        self.lokalt = self.create_lokalt(hidden_channels, reduce_op, lokalt)
        self.globalt = Globalt(
            hidden_channels=hidden_channels,
            reduce_op=reduce_op,
            out_channels=out_channels,
            **globalt
        )
        self.derivative = True
        self.reset_parameters()

    def create_lokalt(self, hidden_channels: int, reduce_op: str, lokalt: dict) -> L.LightningModule:
        return LokaltGradient(
            hidden_channels=hidden_channels,
            reduce_op=reduce_op,
            **lokalt
        )

    def reset_parameters(self):
        self.lokalt.reset_parameters()
        self.globalt.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        tabsopslag = {}
        tabsopslag['lokalt'] = self.lokalt(z, pos, batch, x, v, noise_idx, noise_scale, target)
        tabsopslag['globalt'] = self.globalt(z, pos, batch, x, v, noise_idx, noise_scale)
        return tabsopslag

    @property
    def tabsnøgler(self):
        return ['lokalt', 'globalt']

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
        self.reset_parameters()

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

class Globalt(L.LightningModule):
    def __init__(self,
                 hidden_channels: int = 128,
                 out_channels: int = 4,
                 num_layers: int = 2,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        self.reduce_op = reduce_op
        self.motor = []
        for i in range(num_layers-1):
            self.motor.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.motor.append(torch.nn.ReLU())
        self.motor.append(
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.motor = torch.nn.Sequential(*self.motor)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale) -> torch.Tensor:
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)
        loss = self.criterion(x, noise_idx)
        return loss

    def reset_parameters(self):
        for m in self.motor.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

class GlobaltReg(L.LightningModule):
    def __init__(self,
                 hidden_channels: int = 128,
                 out_channels: int = 1,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        assert out_channels == 1
        self.reduce_op = reduce_op
        self.motor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.criterion = torch.nn.MSELoss()
        self.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale) -> torch.Tensor:
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)

        loss = self.criterion(x, torch.log10(noise_scale))
        if loss > 10**5:
            print(x, torch.log10(noise_scale),  loss.item())
        return loss

    def reset_parameters(self):
        for m in self.motor.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)


class LokalGradientDumt(LokaltGradient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = SimpelMotor(hidden_channels=kwargs['hidden_channels'])

    def reset_parameters(self):
        self.motor.reset_parameters()

class SimpelMotor(L.LightningModule):
    def __init__(self, hidden_channels):
        super().__init__()
        self.motor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, x, v):
        return self.motor(x)

    def reset_parameters(self):
        for layer in self.motor:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)


class HovedSelvvejledtDumt(HovedSelvvejledt):
    def create_lokalt(self, hidden_channels: int, reduce_op: str, lokalt: dict) -> L.LightningModule:
        return LokalGradientDumt(
            hidden_channels=hidden_channels,
            reduce_op=reduce_op,
            **lokalt
        )


