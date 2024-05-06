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
import shutil
from lightning.pytorch.loggers import WandbLogger
import wandb
import subprocess
import sys

LOG_ROOT = "eksp2_logs"

torch.set_float32_matmul_precision('medium')

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 4
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    for opgave in r.get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 5
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8
    # config['udgaver'] = ['med', 'uden']
    # config['temperaturer'] = ['frossen']

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--selvQM9', action='store_true', help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

class Eksp2:
    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        self.args = args
        self.config = src.redskaber.load_config(args.eksp2_path)
        self.selv_ckpt_path = self.config['selv_ckpt_path']
        if args.debug:
            debugify_config(self.config)
        self.init_kørselsid()
        self.fortræn_tags = []
        self.bedste_selvvejledt, self.qm9Bygger2Hoved, self.artefakt_sti, self.run_id = r.get_selvvejledt_fra_wandb(self.config,
                                                                                                                    self.selv_ckpt_path)
        self.fortræn_tags.append(self.run_id)
        for variant in self.config['downstream'].keys():
            self.config['downstream'][variant]['model']['rygrad'] = self.bedste_selvvejledt.args_dict['rygrad']

    def init_kørselsid(self):
        if self.config['kørselsid'] == None:
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
        else:
            self.kørselsid = self.config['kørselsid']


    def get_trainer(self, temperatur, tags=[]):
        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        logger = WandbLogger(project='afhandling', log_model=False, tags=['downstream']+tags,
                             group=f"eksp2_{self.kørselsid}")
        config_curr = self.config['downstream'][temperatur]
        trainer = L.Trainer(max_epochs=config_curr['epoker'],
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=logger,
                            check_val_every_n_epoch=config_curr['check_val_every_n_epoch'],
                            )
        return trainer

    def eftertræn(self, udgave, temperatur):
        assert temperatur in ['frossen', 'optøet']
        assert udgave in ['med', 'uden', 'baseline']
        args_dict = self.config['downstream'][temperatur]['model']
        downstream = m.Downstream(args_dict=args_dict)
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if temperatur == "frossen":
            downstream.frys_rygrad()
        tags = [udgave, temperatur]+self.fortræn_tags
        trainer = self.get_trainer(temperatur, tags=tags)
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()

    def eftertræn_baseline(self):
        args_dict = self.config['downstream']['optøet']['model']
        downstream = m.DownstreamBaselineMean(args_dict=args_dict)
        tags = ['baseline'] + self.fortræn_tags
        trainer = self.get_trainer('optøet', tags=tags)
        trainer.test(model=downstream, datamodule=self.qm9Bygger2Hoved)
        wandb.finish()

    def eksperiment_runde(self, i):
        self.qm9Bygger2Hoved.sample_train_reduced(i)
        for temperatur in self.config['temperaturer']:
            for udgave in self.config['udgaver']:
                self.eftertræn(udgave, temperatur)
        self.eftertræn_baseline()


    def main(self):
        for i in range(self.qm9Bygger2Hoved.n_trin):
            self.eksperiment_runde(i)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
