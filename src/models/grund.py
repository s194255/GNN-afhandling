import lightning as L
from src.models.visnet import VisNetRyggrad
import torch
from typing import Tuple, List, Optional


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
                 rygrad_args: dict,
                 ):
        super().__init__()
        self.selvvejledt = None
        self.tjek_args(args_dict, self.udgangsargsdict)
        self.args_dict = args_dict
        self.tjek_args(rygrad_args, VisNetRyggrad.args)
        self.rygrad = VisNetRyggrad(
            **rygrad_args
        )
        self.hoved = L.LightningModule()
        self.save_hyperparameters()

    def tjek_args(self, givne_args, forventede_args):
        forskel2 = set(forventede_args.keys()) - set(givne_args.keys())
        assert len(forskel2) == 0, f'Følgende argumenter mangler {forskel2}'

    def indæs_selvvejledt_rygrad(self, grundmodel):
        assert grundmodel.hparams.rygrad_args == self.hparams.rygrad_args, 'downstreams rygrad skal bruge samme argumenter som den selvvejledte'
        state_dict = grundmodel.rygrad.state_dict()
        self.rygrad.load_state_dict(state_dict)

    def frys_rygrad(self):
        self.rygrad.freeze()

    @property
    def udgangsargsdict(self):
        # return {"lr": 0.00001, "step_size": 20, "gamma": 0.5}
        return {"lr": 0.00001, "step_size": 20, "gamma": 0.5, "ønsket_lr": 0.001, "opvarmningsperiode": 10}



    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
        lr = self.hparams.args_dict['lr']
        ønsket_lr = self.hparams.args_dict['ønsket_lr']
        opvarmningsperiode = self.hparams.args_dict['opvarmningsperiode']
        gamma = self.hparams.args_dict['gamma']
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        step_size = self.hparams.args_dict['step_size']
        opvarmningsgamma = (ønsket_lr/lr)**(1/opvarmningsperiode)
        scheduler = WarmUpStepLR(optimizer,
                                 step_size=step_size,
                                 gamma=gamma,
                                 opvarmningsgamma=opvarmningsgamma,
                                 opvarmningsperiode=opvarmningsperiode)
        return [optimizer], [scheduler]
