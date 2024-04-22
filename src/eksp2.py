import os.path

import lightning as L
import torchmetrics
import yaml
from lightning.pytorch.loggers import CSVLogger

import src.models as m
import argparse
import torch

import src.models.downstream
import src.models.selvvejledt
from src.data import QM9Bygger2
import src.redskaber as r
import copy
from torch_geometric.data import Data
import pandas as pd
import time
import shutil
from lightning.pytorch.loggers import WandbLogger
import wandb

LOG_ROOT = "eksp2_logs"

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
        self.metric.reset()

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

class Eksp2:
    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        # self.init_kørsel_path()
        self.selv_ckpt_path = args.selv_ckpt_path
        self.config = m.load_config(args.eksp2_path)
        # m.save_config(self.config, os.path.join(self.kørsel_path, "configs.yaml"))
        self.init_df()
        self.init_kørselsid()
        self.init_csv_path()
        if args.selv_ckpt_path:
            api = wandb.Api()
            artefakt = api.artifact(args.selv_ckpt_path)
            artefakt_path = os.path.join(artefakt.download(), 'model.ckpt')
            self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(artefakt_path)
            self.qm9Bygger2Hoved = QM9Bygger2.load_from_checkpoint(artefakt_path, **self.config['datasæt'])
            self.config['rygrad'] = self.bedste_selvvejledt.hparams.rygrad_args
        else:
            self.fortræn()

    def init_df(self):
        self.resultater = {}
        for udgave in self.udgaver:
            for frys in [True, False]:
                for log_metric in self.log_metrics:
                    nøgle = f'{udgave}_{frys}_{log_metric}'
                    self.resultater[nøgle] = []
        self.resultater['datamængde'] = []
        self.resultater['i'] = []
        self.resultater = pd.DataFrame(data=self.resultater)

    def init_csv_path(self):
        if os.path.exists(LOG_ROOT):
            kørsler = os.listdir(LOG_ROOT)
            kørsler = [int(version.split("_")[-1]) for version in kørsler]
            self.csv_id = max(kørsler, default=-1) + 1
        else:
            self.csv_id = 0
        csv_mappe = os.path.join(LOG_ROOT, f"csv_{self.csv_id}")
        os.makedirs(csv_mappe)
        self.csv_path = os.path.join(csv_mappe, 'log_metrics.csv')

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
        logger = WandbLogger(project='afhandling', log_model=log_models[opgave], tags=[opgave, f"csv_id_{self.csv_id}"]+tags,
                             group=f"eksp2_{self.kørselsid}")
        max_epochs = max([trainer_dict['epoker'], epoch])
        trainer = L.Trainer(max_epochs=max_epochs,
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=logger,
                            )
        return trainer
    def fortræn(self):
        selvvejledt = src.models.selvvejledt.Selvvejledt(rygrad_args=self.config['rygrad'],
                                                         hoved_args=self.config['selvvejledt']['hoved'],
                                                         args_dict=self.config['selvvejledt']['model'])
        self.qm9Bygger2Hoved = QM9Bygger2(**self.config['datasæt'])
        epoch = -1
        trainer = self.get_trainer(opgave='selvvejledt', epoch=epoch)
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved, ckpt_path=self.selv_ckpt_path)
        self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        time.sleep(1)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))

    def eftertræn(self, trin, udgave, frys_rygrad):
        frys_rygrad_tags = {
            True: 'frossen',
            False: 'optøet'
        }
        self.qm9Bygger2Hoved.sample_train_reduced(trin)
        downstream = DownstreamEksp2(rygrad_args=self.config['rygrad'],
                                     hoved_args=self.config['downstream']['hoved'],
                                     args_dict=self.config['downstream']['model'])
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if frys_rygrad:
            downstream.frys_rygrad()
        trainer = self.get_trainer('downstream', tags=[udgave, frys_rygrad_tags[frys_rygrad]])
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        resultat = trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)[0]
        time.sleep(1)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()
        return {f'{udgave}_{frys_rygrad}_{log_metric}': [værdi] for log_metric, værdi in resultat.items()}

    def eksperiment_runde(self, i):
        resultat = {}
        for frys_rygrad in [True, False]:
            for udgave in self.udgaver:
                udgave_resultat = self.eftertræn(i, udgave, frys_rygrad)
                resultat = {**resultat, **udgave_resultat}
        resultat['datamængde'] = [self.qm9Bygger2Hoved.get_eftertræningsmængde()]
        resultat['i'] = [i]
        self.resultater = pd.concat([self.resultater, pd.DataFrame(data=resultat)], ignore_index=True)
        self.resultater.to_csv(self.csv_path, index=False)

    def main(self):
        for i in range(len(self.qm9Bygger2Hoved.eftertræningsandele)):
            self.eksperiment_runde(i)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
