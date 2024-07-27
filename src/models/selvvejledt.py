from typing import Tuple, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src import models as m
from src.models.grund import Grundmodel
from src.models.hoveder.hovedselvvejledt import HovedSelvvejledtKlogt, HovedSelvvejledtDumt, HovedSelvvejledtKlogtReg
from src.models.hoveder.hoveddownstream import HovedDownstreamKlogtMD17
from src.models.redskaber import RiemannGaussian, RiemannGaussian2
from torch_geometric.utils import scatter
from lightning.pytorch.utilities import grad_norm


class Selvvejledt(Grundmodel):
    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambdaer = self.args_dict['lambdaer']
        self.register_buffer("noise_scales_options", torch.logspace(
            self.args_dict['noise_fra'],
            self.args_dict['noise_til'],
            steps=self.args_dict['n_noise_trin'],
        ))
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.riemannGaussian = RiemannGaussian()

    def create_hoved(self):
        args = {
            'n_noise_trin': self.args_dict['n_noise_trin'],
            'hidden_channels': self.hidden_channels,
            'beregn_lokalt': self.beregn_lokalt,
            'beregn_globalt': self.beregn_globalt
        }
        args = {**args, **self.args_dict['hoved']}
        if self.args_dict['hovedtype'] == "klogt":
            return HovedSelvvejledtKlogt(**args)
        elif self.args_dict['hovedtype'] == "klogt_reg":
            return HovedSelvvejledtKlogtReg(**args)
        elif self.args_dict['hovedtype'] == "dumt":
            return HovedSelvvejledtDumt(**args)
        else:
            raise NotImplementedError

    def forward(
            self,
            data: Data
    ) -> dict:
        z, pos, batch, edge_index = data.z, data.pos, data.batch, data.edge_index
        noise_idxs = torch.randint(low=0, high=len(self.noise_scales_options),
                                   size=torch.unique(batch).shape, device=self.device)
        noise_scales = torch.gather(self.noise_scales_options, 0, noise_idxs)
        sigma = torch.gather(noise_scales, 0, batch)
        pos_til, target = self.riemannGaussian(pos, batch, sigma)
        if self.hoved.derivative:
            pos_til.requires_grad_(True)
        x, v, _, _ = self.rygrad(z, pos_til, batch, edge_index)
        graph_noisy = {'z': z, 'pos': pos_til, 'batch': batch, 'x': x, 'v': v}
        if self.beregn_globalt:
            x2, v2, _, _ = self.rygrad(z, pos, batch, edge_index)
            graph_normal = {'z': z, 'pos': pos, 'batch': batch, 'x': x2, 'v': v2}
        else:
            graph_normal = None
        tabsopslag = self.hoved(graph_noisy, graph_normal, noise_idxs, noise_scales, target)
        return tabsopslag

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        tabsopslag = self(data)
        tabsopslag = {nøgle: self.lambdaer[nøgle]*værdi for nøgle, værdi in tabsopslag.items()}
        loss = sum([værdi for værdi in tabsopslag.values()])
        log_dict = {"train_loss": loss.item()}
        # self.log("train_loss", loss.item(), batch_size=data.batch_size)
        for nøgle in self.hoved.tabsnøgler:
            log_dict[f"train_{nøgle}_loss"] = tabsopslag[nøgle].item()
            # self.log(f"train_{nøgle}_loss", tabsopslag[nøgle].item(), batch_size=data.batch_size)
        self.log_dict(log_dict, batch_size=data.batch_size)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data)

        tabsopslag = {nøgle: self.lambdaer[nøgle] * værdi for nøgle, værdi in tabsopslag.items()}
        loss = sum([værdi for værdi in tabsopslag.values()])
        log_dict = {"val_loss": loss.item()}
        # self.log("train_loss", loss.item(), batch_size=data.batch_size)
        for nøgle in self.hoved.tabsnøgler:
            log_dict[f"val_{nøgle}_loss"] = tabsopslag[nøgle].item()
            # self.log(f"train_{nøgle}_loss", tabsopslag[nøgle].item(), batch_size=data.batch_size)
        self.log_dict(log_dict, batch_size=data.batch_size)
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("test_loss", loss.item(), batch_size=data.batch_size, on_epoch=True)
        return loss

    @property
    def krævne_args(self):
        nye_args = {'lambdaer', 'noise_fra', 'noise_til', 'n_noise_trin', 'hovedtype', 'hoved'}
        return nye_args.union(super().krævne_args)

    @property
    def selvvejledt(self):
        return True

    def get_fortræningsudgave(self):
        stamme = '3D-EMGP'
        if self.beregn_lokalt and self.beregn_globalt:
            return f'{stamme}-begge'
        elif self.beregn_lokalt and (not self.beregn_globalt):
            return f'{stamme}-lokalt'
        elif (not self.beregn_lokalt) and self.beregn_globalt:
            return f'{stamme}-globalt'
        elif (not self.beregn_lokalt) and (not self.beregn_globalt):
            return 'påskeæg'

    @property
    def beregn_lokalt(self):
        return self.args_dict['lambdaer']['lokalt'] != 0

    @property
    def beregn_globalt(self):
        return self.args_dict['lambdaer']['globalt'] != 0

class HovedPenergi(HovedDownstreamKlogtMD17):
    @property
    def tabsnøgler(self):
        return ['lokalt', 'globalt']
class SelvvejledtPenergi(Selvvejledt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.riemannGaussian = RiemannGaussian2()

    def get_graph_rep(self, z, p, batch, edge_index):
        x, v, _, _ = self.rygrad(z, p, batch, edge_index)
        graph = {'z': z, 'pos': p, 'batch': batch, 'x': x, 'v': v}
        return graph

    def create_hoved(self):
        assert self.args_dict['hovedtype'] == "klogt"
        args = {
            'means': torch.tensor(0),
            'stds': torch.tensor(1),
            'hidden_channels': self.hidden_channels*2,
            'calc_forces': True
        }
        args = {**args, **self.args_dict['hoved']}
        return HovedPenergi(**args)
    def forward(
            self,
            data: Data
    ) -> dict:
        z, pos, batch, edge_index = data.z, data.pos, data.batch, data.edge_index
        noise_idxs = torch.randint(low=0, high=len(self.noise_scales_options),
                                   size=torch.unique(batch).shape, device=self.device)
        noise_scales = torch.gather(self.noise_scales_options, 0, noise_idxs)
        sigma = torch.gather(noise_scales, 0, batch)
        pos_til, force_target, energy_target = self.riemannGaussian(pos, batch, sigma)
        pos_til.requires_grad_(True)

        graph_noisy = self.get_graph_rep(z, pos_til, batch, edge_index)
        graph_normal = self.get_graph_rep(z, pos, batch, edge_index)

        graph_combined = {**{'batch': graph_noisy['batch'], 'z': graph_noisy['z'], 'pos': graph_noisy['pos']},
                          **{key: torch.cat([graph_noisy[key], graph_normal[key]], dim=-1)
                             for key in ['x', 'v']}}

        energy, force = self.hoved(**graph_combined)
        tabsopslag = {
            'lokalt': self.criterion(energy, energy_target),
            'globalt': self.criterion(force, force_target)
        }
        return tabsopslag


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
            data: Data
    ) -> dict:
        z, pos, batch, edge_index = data.z, data.pos, data.batch, data.edge_index
        x, v, edge_attr, masker = self.rygrad(z, pos, batch, edge_index)
        x = x[masker['knuder']]
        z = z[masker['knuder']]
        x = self.hoved(x)
        lokal = self.criterionlokal(x, z)
        globall = torch.tensor(0, device=self.device)
        return {'lokalt': lokal, 'globalt': globall}

    @property
    def krævne_args(self) -> set:
        return super().krævne_args - {'lambdaer'}



class SelvvejledtQM9(m.DownstreamQM9):
    @property
    def selvvejledt(self):
        return True

    @property
    def krævne_args(self) -> set:
        return super().krævne_args - {'frossen_opvarmningsperiode'}

    def get_fortræningsudgave(self):
        return self.__class__.__name__

    def setup(self, stage: str) -> None:
        self.frossen_opv = False