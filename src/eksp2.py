import os.path

import lightning as L
import torchmetrics
from lightning.pytorch.loggers import CSVLogger

import src.models as m
import argparse
import torch

import src.models.downstream
import src.models.selvvejledt
import src.data as d
import src.redskaber
import src.redskaber as r
from torch_geometric.data import Data
import pandas as pd
import shutil
from lightning.pytorch.loggers import WandbLogger
import wandb

LOG_ROOT = "eksp2_logs"

def manip_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    config['downstream']['epoker'] = 1
    config['selvvejledt']['epoker'] = 1
    config['rygrad']['hidden_channels'] = 8

class DownstreamEksp2(src.models.downstream.Downstream):
    def setup(self, stage: str) -> None:
        if stage == 'test':
            self.metric = torchmetrics.BootStrapper(
                torchmetrics.regression.MeanAbsoluteError(),
                num_bootstraps=1000,
                quantile=torch.tensor([0.05, 0.95], device=self.device)
            )

    def test_step(self, data: Data, batch_idx: int) -> None:
        super().test_step(data, batch_idx)
        pred = self(data.z, data.pos, data.batch)
        self.metric.update(1000 * pred, 1000 * data.y[:, self.target_idx])

    def on_test_epoch_end(self) -> None:
        data = self.metric.compute()
        self.log("test_loss_mean", data['mean'])
        self.log("test_loss_std", data['std'])
        self.log("test_loss_lower", data['quantile'][0].item())
        self.log("test_loss_upper", data['quantile'][1].item())
        self.log("eftertræningsmængde", self.get_eftertræningsmængde())
        self.log("trin", self.trainer.datamodule.trin)
        self.metric.reset()

    def get_eftertræningsmængde(self):
        debug = self.trainer.datamodule.debug
        data_split = self.trainer.datamodule.data_splits[debug]['train_reduced']
        return len(data_split)

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

class Eksp2:
    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        # self.init_kørsel_path()
        self.selv_ckpt_path = args.selv_ckpt_path
        self.config = src.redskaber.load_config(args.eksp2_path)
        if args.debug:
            r.debugify_config(self.config)
        # m.save_config(self.config, os.path.join(self.kørsel_path, "configs.yaml"))
        self.init_kørselsid()
        self.fortræn_tags = []
        self.bedste_selvvejledt, self.qm9Bygger2Hoved, _, run_id = r.get_selvvejledt(self.config, args.selv_ckpt_path)
        if run_id:
            self.fortræn_tags.append(run_id)
        if not args.selv_ckpt_path:
            self.fortræn()

    def init_kørselsid(self):
        wandb.login()
        runs = wandb.Api().runs("afhandling")
        kørselsider = []
        for run in runs:
            gruppe = run.group
            if gruppe:
                if gruppe.split("_")[0] == "eksp2":
                    kørselsid = int(gruppe.split("_")[1])
                    kørselsider.append(kørselsid)
        self.kørselsid = max(kørselsider, default=-1)+1


    def get_trainer(self, opgave, tags=[], epoch=-1):
        assert opgave in ['selvvejledt', 'downstream']
        trainer_dict = self.config[opgave]
        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            # r.earlyStopping(trainer_dict['min_delta'], trainer_dict['patience']),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        log_models = {'selvvejledt': True, 'downstream': False}
        logger = WandbLogger(project='afhandling', log_model=log_models[opgave], tags=[opgave]+tags,
                             group=f"eksp2_{self.kørselsid}")
        max_epochs = max([trainer_dict['epoker'], epoch])
        trainer = L.Trainer(max_epochs=max_epochs,
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            precision='16-mixed',
                            logger=logger,
                            )
        return trainer
    def fortræn(self):
        selvvejledt = src.models.selvvejledt.Selvvejledt(rygrad_args=self.config['rygrad'],
                                                         args_dict=self.config['selvvejledt']['model'])
        self.qm9Bygger2Hoved = d.QM9ByggerEksp2(**self.config['datasæt'])
        epoch = -1
        trainer = self.get_trainer(opgave='selvvejledt', epoch=epoch)
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved, ckpt_path=self.selv_ckpt_path)
        self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))

    def eftertræn(self, trin, udgave, frys_rygrad):
        frys_rygrad_tags = {
            True: 'frossen',
            False: 'optøet'
        }
        self.qm9Bygger2Hoved.sample_train_reduced(trin)
        rygrad_args = self.bedste_selvvejledt.hparams.rygrad_args
        downstream = DownstreamEksp2(rygrad_args=rygrad_args,
                                     args_dict=self.config['downstream']['model'])
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if frys_rygrad:
            downstream.frys_rygrad()
        tags = [udgave, frys_rygrad_tags[frys_rygrad], f"trin_{trin}"]+self.fortræn_tags
        trainer = self.get_trainer('downstream', tags=tags)
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        resultat = trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)[0]
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()
        return {f'{udgave}_{frys_rygrad}_{log_metric}': [værdi] for log_metric, værdi in resultat.items()}

    def eksperiment_runde(self, i):
        resultat = {}
        for frys_rygrad in [False]:
            for udgave in self.udgaver:
                udgave_resultat = self.eftertræn(i, udgave, frys_rygrad)
                resultat = {**resultat, **udgave_resultat}
        resultat['datamængde'] = [self.qm9Bygger2Hoved.get_eftertræningsmængde()]
        resultat['i'] = [i]
        # self.resultater = pd.concat([self.resultater, pd.DataFrame(data=resultat)], ignore_index=True)
        # self.resultater.to_csv(self.csv_path, index=False)

    def main(self):
        for i in range(len(self.qm9Bygger2Hoved.eftertræningsandele)):
            self.eksperiment_runde(i)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
