from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.data import get_metadata
from src.models.grund import Grundmodel
from src.models.hoveder.hoveddownstream import HovedDownstream


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
