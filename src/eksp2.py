import os.path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import src.models as m
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

DOWNSTREAMKLASSER = {
    'QM9': m.DownstreamQM9,
    'MD17': m.DownstreamMD17
}

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 4
    config['datasæt']['num_workers'] = 0
    # config['datasæt']['n_trin'] = 2
    config['kørselsid'] = None
    config['gruppenavn'] = 'debug'
    for opgave in r.get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 1
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            # config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8

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
        self.name = self.config['datasæt']['name']
        _, self.qm9Bygger2Hoved, _, _ = r.get_selvvejledt_fra_wandb(self.config, self.config['data_path'])
        if type(self.config['seeds']) == int:
            self.seeds = [None]*self.config['seeds']
        elif type(self.config['seeds']) == list:
            self.seeds = self.config['seeds']
        else:
            raise NotImplementedError

    def init_kørselsid(self) -> None:
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


    def get_trainer(self, config_curr: dict,
                    logger_config: dict=None, tags=None) -> L.Trainer:
        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        logger = WandbLogger(project='afhandling', log_model=False, tags=tags,
                             group=f"{self.config['gruppenavn']}_{self.kørselsid}", config=logger_config)
        trainer = L.Trainer(max_epochs=config_curr['epoker'],
                            max_steps=config_curr['steps'],
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=logger,
                            check_val_every_n_epoch=config_curr['check_val_every_n_epoch'],
                            gradient_clip_val=config_curr['gradient_clipping'],
                            inference_mode=False,
                            )
        return trainer

    def create_downstream(self, udgave: str, lag: int, config_curr: dict) -> Tuple[m.DownstreamQM9, str]:
        args_dict = copy.deepcopy(config_curr['model'])
        metadata = self.qm9Bygger2Hoved.get_metadata('train_reduced')
        if lag is not None:
            print(f'nu sætter jeg et lag = {lag}')
            args_dict['hoved']['num_layers'] = lag

        if udgave != 'uden':
            selvvejledt, qm9bygger, _, run_id = r.get_selvvejledt_fra_wandb(self.config, udgave)
            assert self.qm9Bygger2Hoved.eq_data_split(qm9bygger)
            downstream = DOWNSTREAMKLASSER[self.name](
                args_dict=args_dict,
                metadata=metadata
            )
            downstream.indæs_selvvejledt_rygrad(selvvejledt)
        else:
            downstream = DOWNSTREAMKLASSER[self.name](args_dict=args_dict,
                                         metadata=self.qm9Bygger2Hoved.get_metadata('train_reduced'))
            run_id = None

        return downstream, run_id
    def eftertræn(self, udgave, temperatur, seed, i, lag) -> None:
        assert temperatur in ['frossen', 'optøet']
        if seed != None:
            print(f"jeg planter frøet {seed}")
            torch.manual_seed(seed)
        else:
            print("jeg planter ikke noget frø")
        self.qm9Bygger2Hoved.sample_train_reduced(i)

        config_curr = self.config[self.name][temperatur]
        if config_curr['mixed'] == True:
            torch.set_float32_matmul_precision('medium')
        downstream, run_id = self.create_downstream(udgave=udgave, lag=lag, config_curr=config_curr)
        if temperatur == "frossen":
            downstream.frys_rygrad()
        logger_config = {'fortræningsudgave': downstream.fortræningsudgave,
                         'temperatur': temperatur,
                         'seed': seed,
                         'rygrad runid': run_id,
                         'opgave': 'eftertræn'
                         }
        trainer = self.get_trainer(config_curr=config_curr, logger_config=logger_config)
        trainer.fit(model=downstream, datamodule=self.qm9Bygger2Hoved)
        trainer.test(ckpt_path="best", datamodule=self.qm9Bygger2Hoved)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
        downstream.cpu()

    def eftertræn_baseline(self) -> None:
        args_dict = self.config['Downstream']['optøet']['model']
        downstream = m.DownstreamQM9BaselineMean(
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

    def main(self):

        løkke = product(self.config['temperaturer'], self.seeds, self.config['lag_liste'],
                        range(self.qm9Bygger2Hoved.n_trin), self.config['udgaver'])
        for temperatur, seed, lag, i, udgave in løkke:
            self.eftertræn(udgave=udgave, temperatur=temperatur,
                           seed=seed, i=i, lag=lag)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
