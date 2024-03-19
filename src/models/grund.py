import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.data import QM9Bygger
from src.models.redskaber import Maskemager, RiemannGaussian
import torch
from torch import Tensor
from typing import Tuple, Optional, List
from src.models.hoveder.hovedselvvejledt import HovedSelvvejledt
from src.models.hoveder.hoveddownstream import HovedDownstream
from src.models.visnet import VisNetRyggrad


class Grundmodel(L.LightningModule):
    træn_args = {
        'debug': False,
        'delmængdestørrelse': 0.1,
        'fordeling': None,
    }
    def __init__(self,
                 debug: bool = træn_args['debug'],
                 delmængdestørrelse: float = træn_args['delmængdestørrelse'],
                 fordeling: List = træn_args['fordeling'],
                 rygrad_args=VisNetRyggrad.args
                 ):
        super().__init__()
        self.tjek_args(rygrad_args, VisNetRyggrad.args)
        self.rygrad = VisNetRyggrad(
            **rygrad_args
        )
        self.hoved = L.LightningModule()
        self.debug = debug
        # self.eftertræningsandel = eftertræningsandel
        self.QM9Bygger = QM9Bygger(delmængdestørrelse, fordeling)
        self.save_hyperparameters()

    def tjek_args(self, givne_args, forventede_args):
        forskel1 = set(givne_args.keys()) - set(forventede_args.keys())
        assert len(forskel1) == 0, f'Følgende argumenter var uventede {forskel1}'
        forskel2 = set(forventede_args.keys()) - set(givne_args.keys())
        assert len(forskel2) == 0, f'Følgende argumenter mangler {forskel2}'

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('train', self.debug)

    def val_dataloader(self) -> DataLoader:
        return self.QM9Bygger('val', self.debug)

    def test_dataloader(self) -> DataLoader:
        return self.QM9Bygger('test', self.debug)

    def indæs_selvvejledt_rygrad(self, grundmodel):
        assert grundmodel.hparams.rygrad_args == self.hparams.rygrad_args, 'downstreams rygrad skal bruge samme argumenter som den selvvejledte'
        state_dict = grundmodel.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)

    def frys_rygrad(self):
        self.rygrad.freeze()


class Selvvejledt(Grundmodel):

    def __init__(self, *args,
                 maskeringsandel=0.15,
                 lambdaer=None,
                 hoved_args=HovedSelvvejledt.args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if not lambdaer:
            self.lambdaer = {'lokalt': 0.5, 'globalt': 0.5}
        self.tjek_args(hoved_args, HovedSelvvejledt.args)

        self.register_buffer("noise_scales_options", torch.tensor([0.001, 0.01, 0.1, 1.0, 10, 100, 1000]))
        self.maskeringsandel = maskeringsandel
        self.maskemager = Maskemager()
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

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('pretrain', self.debug)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        # loss = self.criterion(pred, target)
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
        x, v, edge_attr = self.rygrad(z, pos_til, batch)
        tabsopslag = self.hoved(z, pos_til, batch, x, v, noise_idxs, noise_scales, target)
        return tabsopslag


class Downstream(Grundmodel):
    def __init__(self, *args,
                 hoved_args=HovedDownstream.args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tjek_args(hoved_args, HovedDownstream.args)
        self.hoved = HovedDownstream(
            hidden_channels=self.hparams.rygrad_args['hidden_channels'],
            **hoved_args
        )
        self.criterion = torch.nn.L1Loss()

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("val_loss", loss.item(), batch_size=data.batch_size)
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("test_loss", loss.item(), batch_size=data.batch_size, on_epoch=True)
        return loss

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, v, edge_attr = self.rygrad(z, pos, batch)
        y = self.hoved(z, pos, batch, x, v)
        return y

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
