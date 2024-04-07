import os.path

import lightning as L
import torchmetrics
import yaml

import src.models as m
import argparse
import torch
from src.data import QM9Bygger2
from src.redskaber import get_trainer, tensorBoardLogger
import copy
from torch_geometric.data import Data
import pandas as pd
import time

LOG_ROOT = "eksp2_logs"

class DownstreamEksp2(m.Downstream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = torchmetrics.BootStrapper(
            torchmetrics.regression.MeanAbsoluteError(),
            num_bootstraps=20,
            quantile=torch.tensor([0.05, 0.95])
        )

    def test_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        pred = self(data.z, data.pos, data.batch)
        self.metric.update(1000 * pred, 1000 * data.y[:, 0])

    def on_test_epoch_end(self) -> None:
        data = self.metric.compute()
        self.log("test_loss_mean", data['mean'])
        self.log("test_loss_std", data['std'])
        self.log("test_loss_lower", data['quantile'][0].item())
        self.log("test_loss_upper", data['quantile'][1].item())
        self.metric.reset()

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--rygrad_args_path', type=str, default="config/rygrad_args.yaml",
                        help='Sti til rygrad arguments YAML fil')
    parser.add_argument('--selvvejledt_hoved_args_path', type=str, default="config/selvvejledt_hoved_args.yaml",
                        help='Sti til selvvejledt hoved arguments YAML fil')
    parser.add_argument('--downstream_hoved_args_path', type=str, default="config/downstream_hoved_args.yaml",
                        help='Sti til downstream hoved arguments YAML fil')
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--kørsel_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args


class Eksp2:

    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        if args.kørsel_path:
            self.kørsel_path = args.kørsel_path
            self.eftertræningsandele = torch.load(os.path.join(self.kørsel_path, "eftertræningsandele.pth"))
            self.selv_chkt_path = os.path.join(self.kørsel_path, "selvvejledt", "version_0", "checkpoints", "last.ckpt")
            self.configs = m.load_config(os.path.join(self.kørsel_path, "configs.yaml"))
            self.resultater = pd.read_csv(os.path.join(self.kørsel_path, "logs_metrics.csv"))
            self.fra_i = int(self.resultater["i"].max()) + 1
        else:
            self.init_kørsel_path()
            self.selv_chkt_path = args.selv_ckpt_path
            self.configs = {
                'eksp2': m.load_config(args.eksp2_path),
                'rygrad': m.load_config(args.rygrad_args_path),
                'selvvejledt_hoved': m.load_config(args.selvvejledt_hoved_args_path),
                'downstream_hoved': m.load_config(args.downstream_hoved_args_path)
            }
            with open(os.path.join(self.kørsel_path, "configs.yaml"), 'w', encoding='utf-8') as fil:
                yaml.dump(self.configs, fil, allow_unicode=True)
            self.eftertræningsandele = torch.linspace(0.0025, 1.0, steps=self.configs['eksp2']['trin'])
            torch.save(self.eftertræningsandele, os.path.join(self.kørsel_path, 'eftertræningsandele.pth'))
            self.init_resultater()
            self.fra_i = 0

    def init_resultater(self):
        self.resultater = {f'{udgave}_{log_metric}': [] for udgave in self.udgaver for log_metric in self.log_metrics}
        self.resultater['datamængde'] = []
        self.resultater['i'] = []
        self.resultater = pd.DataFrame(data=self.resultater)

    def init_kørsel_path(self):
        if os.path.exists(LOG_ROOT):
            kørsler = os.listdir(LOG_ROOT)
            kørsler = [int(version.split("_")[1]) for version in kørsler]
            kørsel = max(kørsler, default=-1)+1
        else:
            os.mkdir(LOG_ROOT)
            kørsel = 0
        self.kørsel_path = os.path.join(
            LOG_ROOT,
            f'kørsel_{kørsel}'
        )
        os.mkdir(self.kørsel_path)

    def fortræn(self):
        if self.selv_chkt_path:
            selvvejledt = m.Selvvejledt.load_from_checkpoint(self.selv_chkt_path)
            self.qm9Bygger2Hoved = QM9Bygger2.load_from_checkpoint(self.selv_chkt_path)
        else:
            selvvejledt = m.Selvvejledt(rygrad_args=self.configs['rygrad'],
                                    hoved_args=self.configs['selvvejledt_hoved'],
                                    args_dict=self.configs['eksp2']['model'])
            self.qm9Bygger2Hoved = QM9Bygger2(**self.configs['eksp2']['datasæt'],
                                              eftertræningsandel=1.0)
        logger = tensorBoardLogger(save_dir=self.kørsel_path, name='selvvejledt')
        trainer = get_trainer(self.configs['eksp2']['epoker_selvtræn'], logger=logger)
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved, ckpt_path=self.selv_chkt_path)
        self.bedste_selvvejledt = m.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        time.sleep(2)

    def get_qm9Bygger2(self, eftertræningsandel):
        qm9Bygger2 = QM9Bygger2(**self.configs['eksp2']['datasæt'],
                                eftertræningsandel=eftertræningsandel)
        qm9Bygger2.load_state_dict(copy.deepcopy(self.qm9Bygger2Hoved.state_dict()))
        qm9Bygger2.sample_train_reduced()
        return qm9Bygger2

    def eftertræn(self, eftertræningsandel, udgave):
        qm9Bygger2 = self.get_qm9Bygger2(eftertræningsandel)

        downstream = DownstreamEksp2(rygrad_args=self.configs['rygrad'],
                                  hoved_args=self.configs['downstream_hoved'],
                                  args_dict=self.configs['eksp2']['model'])
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if self.configs['eksp2']['frys_rygrad']:
            downstream.frys_rygrad()
        logger = tensorBoardLogger(save_dir=self.kørsel_path, name='efterfølgende')
        trainer = get_trainer(self.configs['eksp2']['epoker_efterfølgende'], logger=logger)
        trainer.fit(model=downstream, datamodule=qm9Bygger2)
        resultat = trainer.test(ckpt_path="best", datamodule=qm9Bygger2)[0]
        time.sleep(2)
        return {f'{udgave}_{nøgle}': [værdi] for nøgle, værdi in resultat.items()}

    def eksperiment_runde(self, i):
        eftertræningsandel = self.eftertræningsandele[i].item()
        resultat = {}
        for udgave in self.udgaver:
            udgave_resultat = self.eftertræn(eftertræningsandel, udgave)
            resultat = {**resultat, **udgave_resultat}
        resultat['datamængde'] = [self.get_datamængde(eftertræningsandel)]
        resultat['i'] = [i]
        self.resultater = pd.concat([self.resultater, pd.DataFrame(data=resultat)], ignore_index=True)
        self.resultater.to_csv(os.path.join(self.kørsel_path, "logs_metrics.csv"), index=False)

    def main(self):
        self.fortræn()
        for i in range(self.fra_i, self.configs['eksp2']['trin']):
            self.eksperiment_runde(i)

    def get_datamængde(self, eftertræningsandel):
        n = len(self.qm9Bygger2Hoved.data_splits[False]['train'])
        return int(n*eftertræningsandel)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
