import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.data import QM9Bygger
from src.models.redskaber import Maskemager, RiemannGaussian
import torch
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from typing import Tuple, Optional
from src.models.hoveder.hovedselvvejledt import HovedSelvvejledt
from src.models.hoveder.hoveddownstream import HovedDownstream

class Grundmodel(L.LightningModule):

    def __init__(self,
                 debug: bool = False,
                 eftertræningsandel: float = 0.0025,
                 delmængdestørrelse: float = 0.1,
                 ):
        super().__init__()
        self.rygrad = L.LightningModule()
        self.hoved = L.LightningModule()
        self.debug = debug
        self.eftertræningsandel = eftertræningsandel
        self.QM9Bygger = QM9Bygger(eftertræningsandel, delmængdestørrelse)

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('train', self.debug)

    def val_dataloader(self) -> DataLoader:
        return self.QM9Bygger('val', self.debug)
    def test_dataloader(self) -> DataLoader:
        return self.QM9Bygger('test', self.debug)

    def indæs_selvvejledt_rygrad(self, grundmodel):
        state_dict = grundmodel.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)

class GrundSelvvejledt(Grundmodel):

    def __init__(self, *args,
                 maskeringsandel = 0.15,
                 lambdaer = None,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 hidden_channels: int = 128,
                 max_z: int = 100,
                 atomref: Optional[Tensor] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if not lambdaer:
            self.lambdaer = {'lokalt': 0.5, 'globalt': 0.5}

        print(self.device)
        self.noise_scales_options = torch.tensor([0.001, 0.01, 0.1, 1.0, 10, 100, 1000], device=self.device)
        self.maskeringsandel = maskeringsandel
        self.maskemager = Maskemager()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.riemannGaussian = RiemannGaussian()
        self.hoved = HovedSelvvejledt(
            atomref=atomref,
            max_z=max_z,
            reduce_op=reduce_op,
            mean=mean,
            std=std,
            hidden_channels=hidden_channels,
            out_channels=len(self.noise_scales_options)
        )

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('pretrain', self.debug)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        # loss = self.criterion(pred, target)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("loss", loss.item(), batch_size=data.batch_size)
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
        print(self.device)
        noise_idxs = torch.randint(low=0, high=len(self.noise_scales_options),
                                   size=torch.unique(batch).shape, device=self.device)
        noise_scales = torch.gather(self.noise_scales_options, 0, noise_idxs)
        sigma = torch.gather(noise_scales, 0, batch)
        pos_til, target = self.riemannGaussian(pos, batch, sigma)
        if self.derivative:
            pos_til.requires_grad_(True)
        edge_index, edge_weight, edge_vec = self.distance(pos_til, batch)
        x, v, edge_attr = self.rygrad(z, pos_til, batch,
                                      edge_index, edge_weight, edge_vec)
        tabsopslag = self.hoved(z, pos_til, batch, x, v, noise_idxs, noise_scales, target)
        return tabsopslag
class GrundDownstream(Grundmodel):
    def __init__(self, *args,
                 hidden_channels: int = 128,
                 out_channels: int = 19,
                 reduce_op: str = 'sum',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hoved = HovedDownstream(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            reduce_op=reduce_op
        )
        self.criterion = torch.nn.MSELoss()

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("loss", loss.item())
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("loss", loss.item())
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred[:, 0], data.y[:, 0])
        self.log("loss", loss.item())
        return loss

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer





