from typing import Tuple, Optional, List

import torch
import torch_geometric.data
from torch import Tensor
from torch_geometric.data import Data

from src.models.grund import Grundmodel
from src.models.hoveder.hoveddownstream import HovedDownstreamKlogt, HovedDownstreamDumt, HovedDownstreamKlogtMD17, GatedEquivariantMotor, PredictRegular
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
        self.criterion = self.create_criterion()
        self.fortræningsudgave = 'uden'
        self.enhedsfaktor = self.create_enhedsfaktor()

    def create_enhedsfaktor(self) -> float:
        raise NotImplementedError

    def create_criterion(self) -> dict:
        criterion = {
            'val': torch.nn.L1Loss(),
            'test': torch.nn.L1Loss()
        }
        if self.args_dict['loss'] == 'L1':
            criterion['train'] = torch.nn.L1Loss()
        elif self.args_dict['loss'] == 'L2':
            criterion['train'] = torch.nn.MSELoss()
        else:
            raise NotImplementedError
        return criterion

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("train", data, batch_idx)

    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        return self.step("val", data, batch_idx)

    @property
    def selvvejledt(self):
        return False

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            if self.args_dict['frossen_opvarmningsperiode'] != None:
                self.frossen_opv = True
                self.frys_rygrad()
            else:
                self.frossen_opv = False
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
        self.fortræningsudgave = selvvejledt.get_fortræningsudgave()
        print(f"domstream rygrad = {self.rygrad_param_sum()}")
        print(f"selvvejledt rygrad = {selvvejledt.rygrad_param_sum()}")

    def test_step(self, data: Data, batch_idx: int) -> None:
        pred = self(data)
        target = self.get_target(data)
        self.metric.update(self.enhedsfaktor * pred, self.enhedsfaktor * target)

    def step(self, task, data, batch_idx):
        on_epoch = {'train': None, 'val': None, 'test': True}
        pred = self(data)
        target = self.get_target(data)
        loss = self.enhedsfaktor*self.criterion[task](pred, target)
        self.log(
            f"{task}_loss", loss.item(),
            batch_size=data.batch_size,
            on_epoch=on_epoch[task],
        )
        return loss

    def get_target(self, data: torch_geometric.data.Data) -> torch.Tensor:
        raise NotImplementedError

    @property
    def krævne_args(self) -> set:
        krævne_args = super().krævne_args
        nye_args = {'loss', 'frossen_opvarmningsperiode'}
        return nye_args.union(krævne_args)

    def on_train_epoch_end(self) -> None:
        if self.frossen_opv:
            if self.trainer.current_epoch == self.args_dict['frossen_opvarmningsperiode']:
                self.tø_rygrad_op()



class DownstreamQM9(Downstream):

    def forward(
            self,
            data: torch_geometric.data.Data
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, v, edge_attr, _ = self.rygrad(data.z, data.pos, data.batch, data.edge_index)
        y = self.hoved(data.z, data.pos, data.batch, x, v)
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

    @property
    def krævne_args(self) -> set:
        nye_args = {"predicted_attribute", "hovedtype", "hoved"}
        return nye_args.union(super().krævne_args)

    def get_target(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return data.y[:, self.target_idx]

    def create_enhedsfaktor(self) -> float:
        return 1000

class DownstreamMD17(Downstream):
    def forward(
            self,
            data: torch_geometric.data.Data
    ) -> Tuple[Tensor, Optional[Tensor]]:
        z, pos, batch, edge_index = data.z, data.pos, data.batch, data.edge_index
        pos.requires_grad_(True)
        x, v, edge_attr, _ = self.rygrad(z, pos, batch, edge_index)
        y = self.hoved(z, pos, batch, x, v)
        return y

    def create_hoved(self):
        assert self.args_dict['hovedtype'] == "klogt"
        args = {
            'means': self.metadata['means'],
            'stds': self.metadata['stds'],
            'hidden_channels': self.hidden_channels
        }
        args = {**args, **self.args_dict['hoved']}
        if self.predicted_attribute == 'force':
            return HovedDownstreamKlogtMD17(**args)
        elif self.predicted_attribute == 'energy':
            return PredictRegular(**args)
        else:
            raise NotImplementedError

    @property
    def krævne_args(self) -> set:
        nye_args = {"predicted_attribute", "hovedtype", "hoved"}
        return nye_args.union(super().krævne_args)

    def get_target(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return data[self.predicted_attribute]
    
    def validation_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            return super().validation_step(data, batch_idx)
            
    def test_step(self, data: Data, batch_idx: int) -> None:
        with torch.enable_grad():
            super().test_step(data, batch_idx)

    def create_enhedsfaktor(self) -> float:
        return 1

    @property
    def predicted_attribute(self):
        return self.args_dict['predicted_attribute']

class DownstreamQM9BaselineMean(DownstreamQM9):

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