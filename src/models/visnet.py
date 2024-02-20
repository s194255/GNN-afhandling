from typing import Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter, subgraph
import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.tg_kilde import Distance
from src.models.redskaber import Maskemager, get_dataloader, VisNetBase


class VisNetSelvvejledt(L.LightningModule):
    def __init__(self, debug: bool):
        super().__init__()
        self.visnetbase = VisNetBase()
        self.edge_out = torch.nn.Linear(self.visnetbase.hidden_channels, 1)
        self.maskeringsandel = 0.15
        self.maskemager = Maskemager()
        self.distance = Distance(self.visnetbase.cutoff,
                                 max_num_neighbors=self.visnetbase.max_num_neighbors)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.debug = debug
        self.save_hyperparameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        masker = self.maskemager(z.shape[0], edge_index, self.maskeringsandel)
        x, edge_attr, edge_index = self.visnetbase(z, pos, batch, edge_index,
                                                   edge_weight, edge_vec, masker)
        edge_attr = self.edge_out(edge_attr)
        edge_attr = edge_attr[masker['kanter']]
        edge_index = edge_index[:, masker['kanter']]
        y = pos[edge_index[0, :], :] - pos[edge_index[1, :], :]
        y = y.square().sum(dim=1, keepdim=True)
        return edge_attr, y

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred, target)
        self.log("loss", loss.item())
        return loss

    def train_dataloader(self) -> DataLoader:
        return get_dataloader('pretrain', self.debug)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred, target)
        self.log("loss", loss.item())
        return loss

    def val_dataloader(self) -> DataLoader:
        return get_dataloader('val', self.debug)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class VisNetDownstream(L.LightningModule):
    def __init__(self, debug: bool,
                 selvvejledt_ckpt: str = None,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 derivative: bool = False,
                 ):
        super().__init__()
        self.init_visnetbase(selvvejledt_ckpt)
        self.edge_out = torch.nn.Linear(self.visnetbase.hidden_channels, 1)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.reduce_op = reduce_op
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.derivative = derivative
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.debug = debug
        self.save_hyperparameters()

    def forward(
            self,
            z: Tensor,
            pos: Tensor,
            batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.derivative:
            pos.requires_grad_(True)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        x, _, _ = self.visnetbase(z, pos, batch, edge_index,
                                  edge_weight, edge_vec)
        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        y = y + self.mean

        if self.derivative:
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
            return y, -dy

        return y, None

    def init_visnetbase(self, selvvejledt_sti: str = None):
        if selvvejledt_sti:
            selvvejledt = VisNetSelvvejledt.load_from_checkpoint(selvvejledt_sti)
            self.visnetbase = selvvejledt.visnetbase
        else:
            self.visnetbase = VisNetBase()

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y, data.y)
        self.log("loss", loss.item())
        return loss

    def train_dataloader(self) -> DataLoader:
        return get_dataloader('train', self.debug)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y, data.y)
        self.log("loss", loss.item())
        return loss

    def val_dataloader(self) -> DataLoader:
        return get_dataloader('val', self.debug)

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y, data.y)
        self.log("loss", loss.item())
        return loss

    def test_dataloader(self) -> DataLoader:
        return get_dataloader('test', self.debug)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer




