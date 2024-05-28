import lightning as L
import src.models.rygrader as rygrader
import torch
from typing import Tuple, List
from lightning.pytorch.utilities import grad_norm


class WarmUpStepLR(torch.optim.lr_scheduler.StepLR):

    def __init__(self, *args,
                 opvarmningsgamma: float,
                 opvarmningsperiode: int,
                 **kwargs):
        self.opvarmningsgamma = opvarmningsgamma
        self.opvarmningsperiode = opvarmningsperiode
        super().__init__(*args, **kwargs)

    def get_lr(self):
        if self.last_epoch > self.opvarmningsperiode:
            return super().get_lr()
        else:
            return [self.optimizer.param_groups[i]['lr'] * self.opvarmningsgamma for i in range(len(self.optimizer.param_groups))]
class Grundmodel(L.LightningModule):
    def __init__(self,
                 args_dict: dict,
                 ):
        super().__init__()
        args_dict.update({"modelklasse": self.__class__.__name__})
        self.args_dict = args_dict
        self.rygrad = self.create_rygrad()
        self.hoved = self.create_hoved()
        self.log_gradient = self.args_dict.get('log_gradient', False)
        self.save_hyperparameters()

    def create_rygrad(self):
        rygrad_args = self.args_dict['rygrad']
        if self.args_dict['rygradtype'] == 'visnet':
            return rygrader.VisNetRygrad(**rygrad_args)
        elif self.args_dict['rygradtype'] == 'baseline':
            return rygrader.BaselineRygrad(**rygrad_args)
        elif self.args_dict['rygradtype'] == 'painn':
            return rygrader.PaiNNRygrad(**rygrad_args)
        else:
            raise NotImplementedError

    @property
    def hidden_channels(self):
        return self.args_dict['rygrad']['hidden_channels']

    def create_hoved(self):
        raise NotImplementedError

    def frys_rygrad(self):
        self.rygrad.freeze()

    def rygrad_param_sum(self):
        total_sum = 0
        for param in self.rygrad.parameters():
            total_sum += param.sum().item()  # Konverter tensor til en enkel værdi
        return total_sum

    @property
    def udgangsargsdict(self):
        # return {"lr": 0.00001, "step_size": 20, "gamma": 0.5}
        return {"lr": 0.00001, "step_size": 20, "gamma": 0.5, "ønsket_lr": 0.001, "opvarmningsperiode": 10}

    @property
    def selvvejledt(self):
        raise NotImplementedError

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.log_gradient:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)



    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        lr = self.hparams.args_dict['lr']
        ønsket_lr = self.hparams.args_dict['ønsket_lr']
        opvarmningsperiode = self.hparams.args_dict['opvarmningsperiode']
        gamma = self.hparams.args_dict['gamma']
        weight_decay = self.hparams.args_dict['weight_decay']
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        step_size = self.hparams.args_dict['step_size']
        opvarmningsgamma = (ønsket_lr/lr)**(1/opvarmningsperiode)
        scheduler = WarmUpStepLR(optimizer,
                                 step_size=step_size,
                                 gamma=gamma,
                                 opvarmningsgamma=opvarmningsgamma,
                                 opvarmningsperiode=opvarmningsperiode)
        return [optimizer], [scheduler]
