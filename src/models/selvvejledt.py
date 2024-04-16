from typing import Tuple, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.models.grund import Grundmodel
from src.models.hoveder.hovedselvvejledt import HovedSelvvejledt
from src.redskaber import RiemannGaussian
from torch_geometric.utils import scatter


class Selvvejledt(Grundmodel):
    def __init__(self, *args,
                 hoved_args=HovedSelvvejledt.args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selvvejledt = True
        if not self.args_dict['lambdaer']:
            lambdaer = {'lokalt': 0.5, 'globalt': 0.5}
        self.lambdaer = lambdaer
        self.tjek_args(hoved_args, HovedSelvvejledt.args)
        self.register_buffer("noise_scales_options", torch.logspace(
            self.hparams.args_dict['noise_fra'],
            self.hparams.args_dict['noise_til'],
            steps=self.hparams.args_dict['n_noise_trin'],
        ))
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.riemannGaussian = RiemannGaussian()
        self.hoved = HovedSelvvejledt(
            **hoved_args,
            hidden_channels=self.hparams.rygrad_args['hidden_channels'],
            out_channels=len(self.noise_scales_options)
        )

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("train_loss", loss.item(), batch_size=data.batch_size)
        for nøgle in self.hoved.tabsnøgler:
            self.log(f"train_{nøgle}_loss", tabsopslag[nøgle].item(), batch_size=data.batch_size)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("val_loss", loss.item(), batch_size=data.batch_size)
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("test_loss", loss.item(), batch_size=data.batch_size, on_epoch=True)
        return loss

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        noise_idxs = torch.randint(low=0, high=len(self.noise_scales_options),
                                   size=torch.unique(batch).shape, device=self.device)
        noise_scales = torch.gather(self.noise_scales_options, 0, noise_idxs)
        sigma = torch.gather(noise_scales, 0, batch)
        pos_til, target = self.riemannGaussian(pos, batch, sigma)
        if self.hoved.derivative:
            pos_til.requires_grad_(True)
        x, v, edge_attr, _ = self.rygrad(z, pos_til, batch)
        tabsopslag = self.hoved(z, pos_til, batch, x, v, noise_idxs, noise_scales, target)
        return tabsopslag

    @property
    def udgangsargsdict(self):
        udgangsargs = {'lambdaer': None,
                       'noise_fra': -3,
                       'noise_til': 3,
                       'n_noise_trin': 4}
        return {**super().udgangsargsdict, **udgangsargs}


class SelvvejledtBaseline(Selvvejledt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         lambdaer={'lokalt': 1.0, 'globalt': 0.0},
                         **kwargs)
        assert self.hparams.rygrad_args['maskeringsandel'] != None
        hidden_channels = self.hparams.rygrad_args['hidden_channels']
        out_channels = self.hparams.rygrad_args['max_z']
        self.hoved = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.criterionlokal = torch.nn.CrossEntropyLoss()

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> dict:

        x, v, edge_attr, masker = self.rygrad(z, pos, batch)
        x = x[masker['knuder']]
        z = z[masker['knuder']]
        x = self.hoved(x)
        lokal = self.criterionlokal(x, z)
        globall = torch.tensor(0, device=self.device)
        return {'lokalt': lokal, 'globalt': globall}

    @property
    def udgangsargsdict(self):
        super_dict = super().udgangsargsdict
        return {key: value for key, value in super_dict.items() if key != 'lambdaer'}

class SelvvejledtContrastive(Grundmodel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_channels = self.hparams.rygrad_args['hidden_channels']
        self.hoved = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 2)
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z: Tensor, pos: Tensor, batch: Tensor,) -> Tuple[Tensor, Tensor]:
        x, v, edge_attr, masker = self.rygrad(z, pos, batch)
        x = scatter(x, batch, dim=0, reduce='sum')
        x = self.hoved(x)
        return x

    def step(self, task: str, data: Data, batch_idx: int):
        on_epoch = {'train': None, 'val': None, 'test': True}
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred, data.y)
        self.log(
            f"{task}_loss", loss.item(),
            batch_size=data.batch_size,
            on_epoch=on_epoch[task],
        )
        return loss
    def training_step(self, data: Data, batch_idx: int):
        return self.step('train', data, batch_idx)

    def validation_step(self, data: Data, batch_idx: int):
        return self.step('val', data, batch_idx)

    def test_step(self, data: Data, batch_idx: int):
        return self.step('test', data, batch_idx)