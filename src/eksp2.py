import os.path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import src.models as m
import src.data as d
import argparse
import torch

import src.models.downstream
import src.models.selvvejledt
import src.redskaber as r
import shutil
from lightning.pytorch.loggers import WandbLogger
import wandb
import copy
from typing import Tuple
from itertools import product

LOG_ROOT = "eksp2_logs"

# torch.set_float32_matmul_precision('medium')

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 4
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    config['kørselsid'] = None
    for opgave in r.get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 1
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

class Eksp2:
    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        self.args = args
        self.config = src.redskaber.load_config(args.eksp2_path)
        if args.debug:
            debugify_config(self.config)
        self.init_kørselsid()
        _, self.qm9Bygger2Hoved, _, _ = r.get_selvvejledt_fra_wandb(self.config, self.config['qm9_path'])

    def init_kørselsid(self):
        if self.config['kørselsid'] == None:
            wandb.login()
            runs = wandb.Api().runs("afhandling")
            kørselsider = []
            for run in runs:
                gruppe = run.group
                if gruppe:
                    if gruppe.split("_")[0] == self.config['gruppenavn']:
                        kørselsid = int(gruppe.split("_")[1])
                        kørselsider.append(kørselsid)
            self.kørselsid = max(kørselsider, default=-1)+1
        else:
            self.kørselsid = self.config['kørselsid']


    def get_trainer(self, temperatur: str, logger_config: dict=None, tags=None):
        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        logger = WandbLogger(project='afhandling', log_model=False, tags=tags,
                             group=f"{self.config['gruppenavn']}_{self.kørselsid}", config=logger_config)
        config_curr = self.config['Downstream'][temperatur]
        trainer = L.Trainer(max_epochs=config_curr['epoker'],
                            max_steps=config_curr['steps'],
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=logger,
                            check_val_every_n_epoch=config_curr['check_val_every_n_epoch'],
                            gradient_clip_val=config_curr['gradient_clipping'],
                            )
        return trainer

    def create_downstream(self, udgave, temperatur) -> Tuple[m.Downstream, str]:
        args_dict = copy.deepcopy(self.config['Downstream'][temperatur]['model'])
        metadata = self.qm9Bygger2Hoved.get_metadata('train_reduced')
        if udgave != 'uden':
            selvvejledt, qm9bygger, _, run_id = r.get_selvvejledt_fra_wandb(self.config, udgave)
            assert self.qm9Bygger2Hoved.eq_data_split(qm9bygger)
            # args_dict['rygrad'] = selvvejledt.args_dict['rygrad']
            downstream = m.Downstream(
                args_dict=args_dict,
                metadata=metadata
            )
            downstream.indæs_selvvejledt_rygrad(selvvejledt)
        else:
            downstream = m.Downstream(args_dict=args_dict,
                                      metadata=self.qm9Bygger2Hoved.get_metadata('train_reduced'))
            run_id = None

        return downstream, run_id
    def eftertræn(self, udgave, temperatur, seed):
        assert temperatur in ['frossen', 'optøet']
        if seed != None:
            print(f"jeg planter frøet {seed}")
            torch.manual_seed(seed)
        else:
            print("jeg planter ikke noget frø")
        downstream, run_id = self.create_downstream(udgave, temperatur)
        if temperatur == "frossen":
            downstream.frys_rygrad()
        logger_config = {'fortræningsudgave': downstream.fortræningsudgave,
                         'temperatur': temperatur,
                         'seed': seed,
                         'rygrad runid': run_id,
                         'opgave': 'eftertræn'
                         }
        trainer = self.get_trainer(temperatur, logger_config=logger_config)
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()

    def eftertræn_baseline(self):
        args_dict = self.config['Downstream']['optøet']['model']
        downstream = m.DownstreamBaselineMean(
            args_dict=args_dict,
            metadata=self.qm9Bygger2Hoved.get_metadata('train_reduced')
        )
        tags = ['baseline']
        logger_config = {'opgave': 'eftertræn',
                         'fortræningsudgave': 'baseline'
                         }
        trainer = self.get_trainer('optøet', tags=tags, logger_config=logger_config)
        trainer.test(model=downstream, datamodule=self.qm9Bygger2Hoved)
        wandb.finish()

    def eksperiment_runde(self, i):
        self.qm9Bygger2Hoved.sample_train_reduced(i)
        for temperatur, seed, udgave in product(self.config['temperaturer'], self.config['seeds'], self.config['udgaver']):
            self.eftertræn(udgave, temperatur, seed)
        if self.config['run_baseline']:
            self.eftertræn_baseline()

    def main(self):
        for i in range(self.qm9Bygger2Hoved.n_trin):
            self.eksperiment_runde(i)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
