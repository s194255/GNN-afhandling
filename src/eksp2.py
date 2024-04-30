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
        config[opgave]['epoker'] = 5
        config[opgave]['check_val_every_n_epoch'] = 1
        config[opgave]['model']['rygrad']['hidden_channels'] = 8
    config['udgaver'] = ['med', 'uden']
    config['temperaturer'] = ['frossen']

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
        # self.init_kørsel_path()
        self.selv_ckpt_path = args.selv_ckpt_path
        self.args = args
        self.config = src.redskaber.load_config(args.eksp2_path)
        if args.debug:
            debugify_config(self.config)
        self.init_kørselsid()
        self.fortræn_tags = []
        modelklasse_str = 'SelvvejledtQM9' if args.selvQM9 else 'Selvvejledt'
        self.bedste_selvvejledt, self.qm9Bygger2Hoved, self.artefakt_sti, self.run_id = r.get_selvvejledt_fra_wandb(self.config, args.selv_ckpt_path,
                                                                                                                    modelklasse_str)
        if self.run_id:
            self.fortræn_tags.append(self.run_id)
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


    def get_trainer(self, opgave, tags=[]):
        assert opgave in ['selvvejledt', 'downstream']
        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            # r.earlyStopping(trainer_dict['min_delta'], trainer_dict['patience']),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        log_models = {'selvvejledt': True, 'downstream': False}
        logger = WandbLogger(project='afhandling', log_model=log_models[opgave], tags=[opgave]+tags,
                             group=f"eksp2_{self.kørselsid}")
        trainer = L.Trainer(max_epochs=self.config[opgave]['epoker'],
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=logger,
                            check_val_every_n_epoch=self.config[opgave]['check_val_every_n_epoch'],
                            )
        return trainer
    def fortræn(self):
        selvvejledt = src.models.selvvejledt.Selvvejledt(args_dict=self.config['selvvejledt']['model'])
        self.qm9Bygger2Hoved = d.QM9ByggerEksp2(**self.config['datasæt'])
        trainer = self.get_trainer(opgave='selvvejledt')
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved, ckpt_path=self.selv_ckpt_path)
        self.artefakt_sti = trainer.checkpoint_callback.best_model_path
        self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(self.artefakt_sti)
        self.run_id = wandb.run.id
        wandb.finish()
        # shutil.rmtree(os.path.join("afhandling", self.run_id))

    def eftertræn(self, udgave, temperatur):
        assert temperatur in ['frossen', 'optøet']
        assert udgave in ['med', 'uden']
        downstream = m.Downstream(args_dict=self.config['downstream']['model'])
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if temperatur == "frossen":
            downstream.frys_rygrad()
        tags = [udgave, temperatur]+self.fortræn_tags
        trainer = self.get_trainer('downstream', tags=tags)
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()

    def eksperiment_runde(self, i):
        self.qm9Bygger2Hoved.sample_train_reduced(i)
        sub_processes = []
        for temperatur in self.config['temperaturer']:
            for udgave in self.config['udgaver']:
                cmd = [
                    sys.executable,
                    "src/train_downstream.py",  # Ændre til den korrekte sti for subprocess-scriptet
                    "--udgave", udgave,
                    "--trin", str(i),
                    "--temperatur", temperatur,
                    "--config_path", self.args.eksp2_path,
                    "--artefakt_sti", self.artefakt_sti,
                    "--run_id", self.run_id,
                    "--kørselsid", str(self.kørselsid),
                    "--debug", str(self.args.debug)
                ]
                process = subprocess.Popen(cmd)
                sub_processes.append(process)
        for process in sub_processes:
            process.wait()

    def main(self):
        for i in range(self.qm9Bygger2Hoved.n_trin):
            self.eksperiment_runde(i)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
