from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.data import get_metadata
from src.models.grund import Grundmodel
from src.models.hoveder.hoveddownstream import PredictDipole, PredictRegular


class Downstream(Grundmodel):
    def __init__(self, *args,
                 hoved_args=PredictDipole.args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selvvejledt = False
        self.tjek_args(hoved_args, PredictDipole.args)
        metadata = get_metadata()
        hoved_args = {
            **hoved_args,
            "means": metadata['means'][self.hparams.args_dict['predicted_attribute']],
            "stds": metadata['stds'][self.hparams.args_dict['predicted_attribute']],
            "hidden_channels": self.hparams.rygrad_args['hidden_channels'],
        }
        self.target_idx = self.hparams.args_dict['predicted_attribute']
        if self.target_idx == 0:
            self.hoved = PredictDipole(**hoved_args, max_z=self.hparams.rygrad_args['max_z'])
        else:
            self.hoved = PredictRegular(**hoved_args)
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
        loss = 1000*self.criterion(pred, data.y[:, self.target_idx])
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

    @property
    def udgangsargsdict(self):
        return {**super().udgangsargsdict, "predicted_attribute": 1}