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

LOG_ROOT = "eksp2_logs"

class DownstreamEksp2(src.models.downstream.Downstream):
    def setup(self, stage: str) -> None:
        if stage == 'test':
            self.metric = torchmetrics.BootStrapper(
                torchmetrics.regression.MeanAbsoluteError(),
                num_bootstraps=100,
                quantile=torch.tensor([0.05, 0.95], device=self.device)
            )

    def test_step(self, data: Data, batch_idx: int) -> None:
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
    parser.add_argument('--eksp2_path', type=str, default="config/eksp2.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--kørsel_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

def get_fordeling(højre_interval):
    n = 130831
    test = 0.04
    train = højre_interval/n*0.8
    val = højre_interval/n*0.2
    pretrain = (1-(test+train+val))*0.8
    preval = (1-(test+train+val))*0.2
    return [pretrain, preval, train, val, test]

class Eksp2:
    def __init__(self, args):
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        if args.kørsel_path:
            self.kørsel_path = args.kørsel_path
            self.eftertræningsandele = torch.load(os.path.join(self.kørsel_path, "eftertræningsandele.pth"))
            self.selv_chkt_path = os.path.join(self.kørsel_path, "selvvejledt", "version_0", "checkpoints", "last.ckpt")
            self.config = m.load_config(os.path.join(self.kørsel_path, "configs.yaml"))
            self.resultater = pd.read_csv(os.path.join(self.kørsel_path, "logs_metrics.csv"))
            self.fra_i = int(self.resultater["i"].max()) + 1
        else:
            self.init_kørsel_path()
            self.selv_chkt_path = args.selv_ckpt_path
            self.config = m.load_config(args.eksp2_path)
            with open(os.path.join(self.kørsel_path, "configs.yaml"), 'w', encoding='utf-8') as fil:
                yaml.dump(self.config, fil, allow_unicode=True)
            self.eftertræningsandele = torch.linspace(self.config['spænd'][0]/130831,
                                                      self.config['spænd'][1]/130831,
                                                      steps=self.config['trin'])
            torch.save(self.eftertræningsandele, os.path.join(self.kørsel_path, 'eftertræningsandele.pth'))
            self.init_resultater()
            self.fra_i = 0

    def init_resultater(self):
        self.resultater = {}
        for udgave in self.udgaver:
            for frys in [True, False]:
                for log_metric in self.log_metrics:
                    nøgle = f'{udgave}_{frys}_{log_metric}'
                    self.resultater[nøgle] = []
        self.resultater['datamængde'] = []
        self.resultater['i'] = []
        self.resultater = pd.DataFrame(data=self.resultater)

    def init_kørsel_path(self):
        a = os.path.join(LOG_ROOT, "logging")
        if os.path.exists(a):
            kørsler = os.listdir(a)
            kørsler = [int(version.split("_")[1]) for version in kørsler]
            kørsel = max(kørsler, default=-1)+1
        else:
            os.makedirs(os.path.join(a))
            kørsel = 0
        self.kørsel_path = os.path.join(
            a,
            f'kørsel_{kørsel}'
        )
        os.makedirs(self.kørsel_path)

    def get_trainer(self, opgave, name, epoch=-1, dirpath=None):
        assert opgave in ['selvvejledt', 'downstream']
        loggers = [
            r.tensorBoardLogger(save_dir=self.kørsel_path, name=name),
        ]
        trainer_dict = self.config[opgave]
        callbacks = [
            r.checkpoint_callback(dirpath=dirpath),
            r.TQDMProgressBar(),
            r.earlyStopping(trainer_dict['min_delta'], trainer_dict['patience']),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        max_epochs = max([trainer_dict['epoker'], epoch])
        trainer = L.Trainer(max_epochs=max_epochs,
                            log_every_n_steps=1,
                            callbacks=callbacks,
                            logger=loggers,
                            )
        return trainer
    def fortræn(self):
        if self.selv_chkt_path:
            self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(self.selv_chkt_path)
            self.qm9Bygger2Hoved = QM9Bygger2.load_from_checkpoint(self.selv_chkt_path,
                                                                   **self.config['datasæt'])
            return

        selvvejledt = src.models.selvvejledt.Selvvejledt(rygrad_args=self.config['rygrad'],
                                                         hoved_args=self.config['selvvejledt']['hoved'],
                                                         args_dict=self.config['selvvejledt']['model'])
        self.qm9Bygger2Hoved = QM9Bygger2(
            **self.config['datasæt'],
            fordeling=get_fordeling(self.config['spænd'][1]),
            eftertræningsandel=1.0
        )
        epoch = -1
        trainer = self.get_trainer(opgave='selvvejledt', epoch=epoch, name="selvvejledt",
                                   dirpath=os.path.join(LOG_ROOT, "checkpoints"))
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved, ckpt_path=self.selv_chkt_path)
        self.bedste_selvvejledt = src.models.selvvejledt.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        time.sleep(3)

    def get_qm9Bygger2(self, eftertræningsandel):
        qm9Bygger2 = QM9Bygger2(**self.config['datasæt'],
                                fordeling=self.qm9Bygger2Hoved.fordeling.tolist(),
                                eftertræningsandel=eftertræningsandel)
        qm9Bygger2.load_state_dict(copy.deepcopy(self.qm9Bygger2Hoved.state_dict()))
        qm9Bygger2.sample_train_reduced()
        return qm9Bygger2

    def eftertræn(self, eftertræningsandel, udgave, frys_rygrad):
        qm9Bygger2 = self.get_qm9Bygger2(eftertræningsandel)

        downstream = DownstreamEksp2(rygrad_args=self.config['rygrad'],
                                     hoved_args=self.config['downstream']['hoved'],
                                     args_dict=self.config['downstream']['model'])
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if frys_rygrad:
            downstream.frys_rygrad()
        trainer = self.get_trainer('downstream', name=f'downstream_{udgave}_{frys_rygrad}')
        trainer.fit(model=downstream, datamodule=qm9Bygger2)
        resultat = trainer.test(ckpt_path="best", datamodule=qm9Bygger2)[0]
        time.sleep(1)
        shutil.rmtree(os.path.join(trainer.log_dir, "checkpoints"))
        return {f'{udgave}_{frys_rygrad}_{log_metric}': [værdi] for log_metric, værdi in resultat.items()}

    def eksperiment_runde(self, i):
        eftertræningsandel = self.eftertræningsandele[i].item()
        resultat = {}
        for udgave in self.udgaver:
            for frys_rygrad in [True, False]:
                udgave_resultat = self.eftertræn(eftertræningsandel, udgave, frys_rygrad)
                resultat = {**resultat, **udgave_resultat}
        resultat['datamængde'] = [self.get_datamængde(eftertræningsandel)]
        resultat['i'] = [i]
        self.resultater = pd.concat([self.resultater, pd.DataFrame(data=resultat)], ignore_index=True)
        self.resultater.to_csv(os.path.join(self.kørsel_path, "logs_metrics.csv"), index=False)

    def main(self):
        self.fortræn()
        for i in range(self.fra_i, self.config['trin']):
            self.eksperiment_runde(i)

    def get_datamængde(self, eftertræningsandel):
        n = len(self.qm9Bygger2Hoved.data_splits[False]['train'])
        return int(n*eftertræningsandel)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
