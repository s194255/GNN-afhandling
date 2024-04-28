from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.data import get_metadata
from src.models.grund import Grundmodel
from src.models.hoveder.hoveddownstream import HovedDownstreamKlogt, HovedDownstreamDumt


class Downstream(Grundmodel):
    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.target_idx = self.hparams.args_dict['predicted_attribute']
        self.hoved = self.create_hoved()
        self.criterion = torch.nn.L1Loss()

    def create_hoved(self):
        metadata = get_metadata()
        if self.args_dict['hovedtype'] == "klogt":
            return HovedDownstreamKlogt(
                **self.args_dict['hoved'],
                means=metadata['means'][self.target_idx],
                stds=metadata['stds'][self.target_idx],
                hidden_channels=self.hparams.rygrad_args['hidden_channels'],
                target_idx=self.target_idx,
                max_z=self.hparams.rygrad_args['max_z'],
            )
        elif self.args_dict['hovedtype'] == "dumt":
            return HovedDownstreamDumt(
                **self.args_dict['hoved'],
                means=metadata['means'][self.target_idx],
                stds=metadata['stds'][self.target_idx],
                hidden_channels=self.hparams.rygrad_args['hidden_channels'],
            )

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

    @property
    def selvvejledt(self):
        return False