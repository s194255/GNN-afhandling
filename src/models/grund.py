import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.data import QM9Bygger
from src.models.redskaber import Maskemager, RiemannGaussian
import torch
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from typing import Tuple

class Grundmodel(L.LightningModule):

    def __init__(self,
                 debug: bool = False,
                 eftertræningsandel: float = 0.0025
                 ):
        super().__init__()
        self.rygrad = L.LightningModule()
        self.hoved = L.LightningModule()
        self.debug = debug
        self.eftertræningsandel = eftertræningsandel
        self.QM9Bygger = QM9Bygger(eftertræningsandel)

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('train', self.debug)

    def val_dataloader(self) -> DataLoader:
        return self.QM9Bygger('val', self.debug)
    def test_dataloader(self) -> DataLoader:
        return self.QM9Bygger('test', self.debug)

    def indæs_selvvejledt_rygrad(self, grundmodel):
        state_dict = grundmodel.motor.state_dict()
        self.rygrad.load_state_dict(state_dict)

class GrundSelvvejledt(Grundmodel):

    def __init__(self, *args,
                 maskeringsandel = 0.15,
                 lambdaer = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not lambdaer:
            self.lambdaer = {'lokalt': 0.5, 'globalt': 0.5}
        self.maskeringsandel = maskeringsandel
        self.maskemager = Maskemager()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.riemannGaussian = RiemannGaussian()
        self.noise_scales_options = torch.tensor([0.001, 0.01, 0.1, 1.0, 10, 100, 1000], device=self.device)

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        tabsopslag = self(data.z, data.pos, data.batch)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("loss", loss.item(), batch_size=data.batch_size)
        print(loss.item())
        return loss

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('pretrain', self.debug)

    # def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
    #     with torch.enable_grad():
    #         pred, target = self(data.z, data.pos, data.batch)
    #     loss = self.criterion(pred, target)
    #     self.log("loss", loss.item(), batch_size=data.batch_size)
    #     print(loss.item())
    #     return loss
    #
    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            tabsopslag = self(data.z, data.pos, data.batch)
        # loss = self.criterion(pred, target)
        loss = sum(self.lambdaer[tab] * tabsopslag[tab] for tab in tabsopslag.keys())
        self.log("loss", loss.item(), batch_size=data.batch_size)
        # print(pred)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.rygrad, norm_type=2)
        norms_list = []
        for norm in norms.values():
            norms_list.append(norm.item())
        print(f"største gradientværdi = {max(norms_list)}")
        print(f"mindste gradientværdi = {min(norms_list)}")

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        noise_idxs = torch.randint(low=0, high=len(self.noise_scales_options), size=torch.unique(batch).shape, device=self.device)
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


