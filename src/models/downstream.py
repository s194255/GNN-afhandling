from typing import Tuple, Optional, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from src.models.grund import Grundmodel
from src.models.hoveder.hoveddownstream import HovedDownstreamKlogt, HovedDownstreamDumt
from src.data.QM9 import QM9ByggerEksp2
import torchmetrics
import lightning as L
import copy


class Downstream(Grundmodel):
    def __init__(self, *args,
                 metadata: dict,
                 **kwargs):
        self.metadata = metadata
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.L1Loss()
        self.fortræningsudgave = 'uden'

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
    def target_idx(self):
        return self.args_dict['predicted_attribute']

    def create_hoved(self):
        # metadata = get_metadata()
        if self.args_dict['hovedtype'] == "klogt":
            return HovedDownstreamKlogt(
                **self.args_dict['hoved'],
                means=self.metadata['means'][self.target_idx],
                stds=self.metadata['stds'][self.target_idx],
                hidden_channels=self.hidden_channels,
                target_idx=self.target_idx,
                max_z=self.args_dict['rygrad']['max_z'],
            )
        elif self.args_dict['hovedtype'] == "dumt":
            return HovedDownstreamDumt(
                **self.args_dict['hoved'],
                means=self.metadata['means'][self.target_idx],
                stds=self.metadata['stds'][self.target_idx],
                hidden_channels=self.hidden_channels,
            )

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("train", data, batch_idx)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("val", data, batch_idx)

    def test_step(self, data: Data, batch_idx: int) -> None:
        pred = self(data.z, data.pos, data.batch)
        self.metric.update(1000 * pred, 1000 * data.y[:, self.target_idx])

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

    @property
    def udgangsargsdict(self):
        return {**super().udgangsargsdict, "predicted_attribute": 1}

    @property
    def selvvejledt(self):
        return False

    def setup(self, stage: str) -> None:
        if stage == 'test':
            self.metric = torchmetrics.BootStrapper(
                torchmetrics.regression.MeanAbsoluteError(),
                num_bootstraps=1000,
                quantile=torch.tensor([0.05, 0.95], device=self.device)
            )

    def on_test_epoch_end(self) -> None:
        data = self.metric.compute()
        log_dict = {
            "test_loss_mean": data['mean'],
            "test_loss_std": data['std'],
            "test_loss_lower": data['quantile'][0].item(),
            "test_loss_upper": data['quantile'][1].item(),
            "eftertræningsmængde": self.get_eftertræningsmængde(),
        }
        self.log_dict(log_dict)
        self.metric.reset()

    def get_eftertræningsmængde(self):
        debug = self.trainer.datamodule.debug
        task = 'train_reduced' if self.trainer.datamodule.__class__ == QM9ByggerEksp2 else 'train'
        data_split = self.trainer.datamodule.data_splits[debug][task]
        return len(data_split)

    def indæs_selvvejledt_rygrad(self, selvvejledt):
        assert self.args_dict['rygradtype'] == selvvejledt.args_dict['rygradtype'], 'downstreams rygradtype skal være det samme som den selvvejledte'
        assert self.args_dict['rygrad'] == selvvejledt.args_dict['rygrad'], 'downstreams rygrad skal bruge samme argumenter som den selvvejledte'
        state_dict = selvvejledt.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)
        self.fortræningsudgave = selvvejledt.__class__.__name__
        print(f"domstream rygrad = {self.rygrad_param_sum()}")
        print(f"selvvejledt rygrad = {selvvejledt.rygrad_param_sum()}")

class DownstreamBaselineMean(Downstream):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mean", copy.deepcopy(self.metadata['means'][self.target_idx]))

    def create_rygrad(self):
        return L.LightningModule()

    def create_hoved(self):
        return L.LightningModule()

    def forward(self,
                z: Tensor,
                pos: Tensor,
                batch: Tensor):
        batch_size = len(torch.unique(batch))
        return self.mean*torch.ones(size=(batch_size,), device=self.device)