from typing import Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.autograd import grad
from torch_geometric.utils import scatter, subgraph
import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.tg_kilde import Distance, ViSNetBlock, EquivariantScalar
from src.models.redskaber import Maskemager
from src.data import QM9Bygger

class VisNetBase(L.LightningModule):
    def __init__(self, debug: bool,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 eftertræningsandel: float = 0.0025
                 ):
        super().__init__()
        self.rygrad = ViSNetBlock()
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.QM9Bygger = QM9Bygger(eftertræningsandel)
        self.debug = debug
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('train', self.debug)

    def val_dataloader(self) -> DataLoader:
        return self.QM9Bygger('val', self.debug)
    def test_dataloader(self) -> DataLoader:
        return self.QM9Bygger('test', self.debug)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def indæs_selvvejledt_rygrad(self, visetbase):
        state_dict = visetbase.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)

class VisNetSelvvejledt(VisNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hoved = torch.nn.Linear(self.rygrad.hidden_channels, 1)
        self.maskeringsandel = 0.15
        self.maskemager = Maskemager()

        self.save_hyperparameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        masker = self.maskemager(z.shape[0], edge_index, self.maskeringsandel)
        x, v, edge_attr = self.rygrad(z, pos, batch,
                                      edge_index, edge_weight, edge_vec)
        edge_attr = self.hoved(edge_attr)
        edge_attr = edge_attr[masker['kanter']]
        edge_index = edge_index[:, masker['kanter']]
        y = pos[edge_index[0, :], :] - pos[edge_index[1, :], :]
        y = y.square().sum(dim=1, keepdim=True)
        return edge_attr, y

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred, target)
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss

    def train_dataloader(self) -> DataLoader:
        return self.QM9Bygger('pretrain', self.debug)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred, target = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred, target)
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss


class VisNetDownstream(VisNetBase):
    def __init__(self, *args,
                 reduce_op: str = "sum",
                 mean: float = 0.0,
                 std: float = 1.0,
                 derivative: bool = False,
                 hidden_channels: int = 128,
                 out_channels: int = 19,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.hoved = EquivariantScalar(hidden_channels=hidden_channels, out_channels=out_channels)
        self.reduce_op = reduce_op
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.derivative = derivative
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
        x, v, edge_attr = self.rygrad(z, pos, batch,
                                      edge_index, edge_weight, edge_vec)
        x = self.hoved(x, v)
        x = x * self.std
        # GØRE: EVENTUELT REIMPLEMENTÉR PRIORMODEL
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

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y[:, 1], data.y[:, 1])
        self.log("loss", loss.item(), batch_size=data.batch_size)
        return loss

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y[:, 1], data.y[:, 1])
        self.log("loss", loss.item(), batch_size=data.batch_size, on_epoch=True)
        return loss

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred_y, pred_dy = self(data.z, data.pos, data.batch)
        loss = self.criterion(pred_y[:, 1], data.y[:, 1])
        self.log("MSE", loss.item(), batch_size=data.batch_size, on_epoch=True)
        return loss





