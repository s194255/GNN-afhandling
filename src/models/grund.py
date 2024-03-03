import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.data import QM9Bygger
from src.models.redskaber import Maskemager
import torch
from lightning.pytorch.utilities import grad_norm

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
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.maskeringsandel = maskeringsandel
        self.maskemager = Maskemager()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = torch.nn.L1Loss(reduction='mean')

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        loss = self(data.z, data.pos, data.batch)
        # loss = self.criterion(pred, target)
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
            loss = self(data.z, data.pos, data.batch)
        # loss = self.criterion(pred, target)
        self.log("loss", loss.item(), batch_size=data.batch_size)
        # print(pred)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.rygrad, norm_type=2)
        norms_list = []
        for norm in norms.values():
            norms_list.append(norm.item())
        print(max(norms_list))
        print(min(norms_list))
