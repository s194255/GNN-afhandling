import lightning as L
from torch_geometric.data import Data
from src.data import  get_metadata
from src.models.redskaber import RiemannGaussian
import torch
from torch import Tensor
from typing import Tuple, Optional, List
from src.models.hoveder.hovedselvvejledt import HovedSelvvejledt
from src.models.hoveder.hoveddownstream import HovedDownstream
from src.models.visnet import VisNetRyggrad
from src.models.redskaber import tjek_args, prune_args


class Grundmodel(L.LightningModule):
    def __init__(self,
                 args_dict: dict,
                 rygrad_args: dict,
                 ):
        super().__init__()
        self.selvvejledt = None
        args_dict = prune_args(args_dict, self.udgangsargsdict)
        tjek_args(args_dict, self.udgangsargsdict)
        self.træn_args = args_dict
        self.tjek_args(rygrad_args, VisNetRyggrad.args)
        self.rygrad = VisNetRyggrad(
            **rygrad_args
        )
        self.hoved = L.LightningModule()
        self.save_hyperparameters()

    def tjek_args(self, givne_args, forventede_args):
        forskel1 = set(givne_args.keys()) - set(forventede_args.keys())
        assert len(forskel1) == 0, f'Følgende argumenter var uventede {forskel1}'
        forskel2 = set(forventede_args.keys()) - set(givne_args.keys())
        assert len(forskel2) == 0, f'Følgende argumenter mangler {forskel2}'

    def indæs_selvvejledt_rygrad(self, grundmodel):
        assert grundmodel.hparams.rygrad_args == self.hparams.rygrad_args, 'downstreams rygrad skal bruge samme argumenter som den selvvejledte'
        state_dict = grundmodel.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)

    def frys_rygrad(self):
        self.rygrad.freeze()

    @property
    def udgangsargsdict(self):
        return {}


class Selvvejledt(Grundmodel):
    # _selvvejledt_args = {'lambdaer': None}
    # udgngsargs = {**Grundmodel.udgngsargs, **_selvvejledt_args}
    def __init__(self, *args,
                 hoved_args=HovedSelvvejledt.args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selvvejledt = True
        if not self.træn_args['lambdaer']:
            lambdaer = {'lokalt': 0.5, 'globalt': 0.5}
        self.lambdaer = lambdaer
        self.tjek_args(hoved_args, HovedSelvvejledt.args)

        self.register_buffer("noise_scales_options", torch.tensor([0.001, 0.01, 0.1, 1.0, 10, 100, 1000]))
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer

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
        udgangsargs = {'lambdaer': None}
        return {**super().udgangsargsdict, **udgangsargs}

class Downstream(Grundmodel):
    def __init__(self, *args,
                 hoved_args=HovedDownstream.args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selvvejledt = False
        self.tjek_args(hoved_args, HovedDownstream.args)
        metadata = get_metadata()
        self.hoved = HovedDownstream(
            means=metadata['means'],
            stds=metadata['stds'],
            hidden_channels=self.hparams.rygrad_args['hidden_channels'],
            max_z=self.hparams.rygrad_args['max_z'],
            **hoved_args
        )
        self.criterion = torch.nn.L1Loss()

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("train", data, batch_idx)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("val", data, batch_idx)

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("test", data, batch_idx)

    def step(self, task, data, batch_idx):
        on_epoch = {'train': None, 'val': None, 'test': True}
        pred = self(data.z, data.pos, data.batch)
        loss = 1000*self.criterion(pred, data.y[:, 0])
        self.log(
            f"{task}_loss", loss.item(),
            batch_size=data.batch_size,
            on_epoch=on_epoch[task],
        )
        return loss

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, v, edge_attr, _ = self.rygrad(z, pos, batch)
        y = self.hoved(z, pos, batch, x, v)
        return y

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args_dict['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.args_dict['step_size'],
                                                    gamma=self.hparams.args_dict['gamma'])
        return [optimizer], [scheduler]

    @property
    def udgangsargsdict(self):
        downstream_args = {"lr": 0.00001, "step_size": 20, "gamma": 0.5}
        return {**super().udgangsargsdict, **downstream_args}

class Selvvejledt2(Selvvejledt):
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

