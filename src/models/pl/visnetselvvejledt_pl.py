import lightning as L
from src.models.visnet import VisNetBase, VisNetDownstream, VisNetSelvvejledt
from src.data.QM9 import QM9LOL, byg_QM9
from typing import Any, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import torch_geometric

class VisNetSelvvejledtPL(L.LightningModule):

    def __init__(self, visnetselvvejledt: VisNetSelvvejledt, debug: bool):
        super().__init__()
        self.hovedmodel = visnetselvvejledt
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.debug = debug

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        edge_attr, y = self.hovedmodel(z, pos, batch)
        return edge_attr, y

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        return self.criterion(pred, target)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('pretrain')

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        return self.criterion(pred, target)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def get_dataloader(self, task: str) -> DataLoader:
        shuffle_options = {'pretrain': True, 'train': True, 'val': False, 'test': False}
        dataset = byg_QM9("data/QM9", task)
        if self.debug:
            subset_indices = random.sample(list(range(len(dataset))), k=50)
            dataset = torch.utils.data.Subset(dataset, subset_indices)
        dataloader = DataLoader(dataset, batch_size=512,
                                shuffle=shuffle_options[task], num_workers=0)
        return dataloader


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer