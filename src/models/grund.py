import lightning as L
from src.models.visnet import VisNetRyggrad
from src.models.redskaber import tjek_args, prune_args
import torch
from typing import Tuple, List, Optional


class WarmUpStepLR(torch.optim.lr_scheduler.StepLR):

    def __init__(self, *args,
                 ønsket_lr: float,
                 epoker: int,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, epoch: Optional[int] = ...) -> None:
        pass
class Grundmodel(L.LightningModule):
    def __init__(self,
                 args_dict: dict,
                 rygrad_args: dict,
                 ):
        super().__init__()
        self.selvvejledt = None
        args_dict = prune_args(args_dict, self.udgangsargsdict)
        tjek_args(args_dict, self.udgangsargsdict)
        self.args_dict = args_dict
        self.tjek_args(rygrad_args, VisNetRyggrad.args)
        self.rygrad = VisNetRyggrad(
            **rygrad_args
        )
        self.hoved = L.LightningModule()
        self.save_hyperparameters()

    def tjek_args(self, givne_args, forventede_args):
        forskel1 = set(givne_args.keys()) - set(forventede_args.keys())
        assert len(forskel1) == 0, f'Følgende argumenter var uventede {forskel1}'
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
        return {"lr": 0.00001, "step_size": 20, "gamma": 0.5}


    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
        lr = self.hparams.args_dict['lr']
        max_lr = self.hparams.args_dict['max_lr']
        warmup_period = self.hparams.args_dict['warmup_period']
        gamma = self.hparams.args_dict['lr']
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        step_size = self.hparams.args_dict['step_size']
        # scheduler1 = torch.optim.lr_scheduler.ChainedScheduler([
        #     torch.optim.lr_scheduler.ConstantLR(optimizer, factor=lr/max_lr),
        #     torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(max_lr/lr)**(1/warmup_period))
        # ])
        # # scheduler1 = torch.optim.lr_scheduler.StepLR(
        # #     optimizer,
        # #     step_size=1,
        # #     gamma=(max_lr/lr)**(1/warmup_period)
        # # )
        # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_period]
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.args_dict['step_size'],
                                                    gamma=self.hparams.args_dict['gamma'])
        def lol(epoch):
            if epoch < warmup_period:
                return (epoch + 1) / warmup_period * (max_lr/lr)**(1/warmup_period)
            else:
                return 1
        scheduler.step = lol
        return [optimizer], [scheduler]
