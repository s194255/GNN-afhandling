from typing import Optional

import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from src.models.hoveder.fælles import GatedEquivariantMotor, LinearMotor


class HovedSelvvejledtKlogt(L.LightningModule):
    def __init__(self,
                 n_noise_trin: int,
                 hidden_channels: int,
                 num_layers: int,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        self.n_noise_trin = n_noise_trin
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.reduce_op = reduce_op


        self.motor = GatedEquivariantMotor(
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            means=torch.zeros(size=(self.out_channels,)),
            stds=torch.ones(size=(self.out_channels,)),
            num_layers=self.num_layers,
            reduce_op=self.reduce_op
        )
        self.derivative = True
        self.criterion_globalt = torch.nn.CrossEntropyLoss()
        self.criterion_lokalt = torch.nn.MSELoss(reduction='none')
        self.reset_parameters()

    def reset_parameters(self):
        self.motor.reset_parameters()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        x = self.motor(z, pos, batch, x, v)
        pred_globalt, pred_lokalt = torch.split(x, self.n_noise_trin, dim=1)
        tabsopslag = {}
        tabsopslag['lokalt'] = self.get_loss_lokalt(z, pos, batch, pred_lokalt, noise_scale, target)
        tabsopslag['globalt'] = self.criterion_globalt(pred_globalt, noise_idx)
        return tabsopslag

    def get_loss_lokalt(self, z, pos, batch, pred_lokalt, noise_scale, target):
        grad_outputs = [torch.ones_like(pred_lokalt)]
        dy = grad(
            [pred_lokalt],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if dy is None:
            raise RuntimeError(
                "Autograd returned None for the force prediction.")
        noise_scale = torch.gather(noise_scale, 0, batch)
        loss = noise_scale ** 2 * self.criterion_lokalt(1 / noise_scale.view(-1, 1) * dy, target).sum(dim=1)
        loss = loss.mean()
        return loss

    @property
    def tabsnøgler(self):
        return ['lokalt', 'globalt']

    @property
    def out_channels(self):
        return self.n_noise_trin+1

class HovedSelvvejledtKlogtReg(HovedSelvvejledtKlogt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion_globalt = torch.nn.MSELoss()

    def forward(self, z, pos, batch, x, v, noise_idx, noise_scale, target):
        x = self.motor(z, pos, batch, x, v)
        pred_globalt, pred_lokalt = torch.split(x, 1, dim=1)
        tabsopslag = {}
        tabsopslag['lokalt'] = self.get_loss_lokalt(z, pos, batch, pred_lokalt, noise_scale, target)
        tabsopslag['globalt'] = self.criterion_globalt(pred_globalt.squeeze(0), torch.log10(noise_scale))
        return tabsopslag

    @property
    def out_channels(self):
        return 2



class HovedSelvvejledtDumt(HovedSelvvejledtKlogt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


