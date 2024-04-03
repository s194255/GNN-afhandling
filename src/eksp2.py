import os.path

import lightning as L
import torchmetrics

import src.models as m
import argparse
import torch
from src.data import QM9Bygger2
from src.redskaber import TQDMProgressBar, checkpoint_callback, get_trainer, tensorBoardLogger
import copy
from torch_geometric.data import Data
import pandas as pd
import time

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
    args = parser.parse_args()
    return args


class Eksp2:

    def __init__(self, args):
        self.args = args
        self.eksp2 = m.load_config(args.eksp2_path)
        self.eftertræningsandele = torch.linspace(0.0025, 1.0, steps=self.eksp2['trin'])
        self.log_metrics = ['test_loss_std', 'test_loss_mean', 'test_loss_lower', 'test_loss_upper']
        self.udgaver = ['uden', 'med']
        self.resultater = {f'{udgave}_{log_metric}': [] for udgave in self.udgaver for log_metric in self.log_metrics}
        self.resultater['datamængde'] = []
        self.init_logdir()

    def init_logdir(self):
        self.log_root = "eksp2_logs"
        if os.path.exists(self.log_root):
            kørsler = os.listdir(self.log_root)
            kørsler = [int(version.split("_")[1]) for version in kørsler]
            kørsel = max(kørsler, default=-1)+1
        else:
            os.mkdir(self.log_root)
            kørsel = 0
        self.save_dir = os.path.join(
            self.log_root,
            f'kørsel_{kørsel}'
        )



    def fortræn(self):
        if self.args.selv_ckpt_path:
            selvvejledt = m.Selvvejledt.load_from_checkpoint(self.args.selv_ckpt_path)
            self.qm9Bygger2Hoved = QM9Bygger2.load_from_checkpoint(self.args.selv_ckpt_path)
        else:
            selvvejledt = m.Selvvejledt(rygrad_args=m.load_config(self.args.rygrad_args_path),
                                    hoved_args=m.load_config(self.args.selvvejledt_hoved_args_path),
                                    træn_args=m.load_config(self.args.eksp2_path, m.Selvvejledt.udgngs_træn_args))
            self.qm9Bygger2Hoved = QM9Bygger2(**m.load_config(self.args.eksp2_path, QM9Bygger2.args),
                                              eftertræningsandel=1.0)
        logger = tensorBoardLogger(save_dir=self.save_dir, name='selvvejledt')
        trainer = get_trainer(self.eksp2['epoker_selvtræn'], logger=logger)
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger2Hoved)
        self.bedste_selvvejledt = m.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    def get_qm9Bygger2(self, eftertræningsandel):
        qm9Bygger2 = QM9Bygger2(**m.load_config(self.args.eksp2_path, QM9Bygger2.args),
                                eftertræningsandel=eftertræningsandel)
        qm9Bygger2.load_state_dict(copy.deepcopy(self.qm9Bygger2Hoved.state_dict()))
        qm9Bygger2.sample_train_reduced()
        return qm9Bygger2

    def eftertræn(self, eftertræningsandel, udgave):
        qm9Bygger2 = self.get_qm9Bygger2(eftertræningsandel)

        downstream = DownstreamEksp2(rygrad_args=m.load_config(self.args.rygrad_args_path),
                                  hoved_args=m.load_config(self.args.downstream_hoved_args_path),
                                  træn_args=m.load_config(self.args.eksp2_path, m.Downstream.udgngs_træn_args))
        if udgave == 'med':
            downstream.indæs_selvvejledt_rygrad(self.bedste_selvvejledt)
        if self.eksp2['frys_rygrad']:
            downstream.frys_rygrad()
        logger = tensorBoardLogger(save_dir=self.save_dir, name='efterfølgende')
        trainer = get_trainer(self.eksp2['epoker_efterfølgende'], logger=logger)
        trainer.fit(model=downstream, datamodule=qm9Bygger2)
        resultater = trainer.test(ckpt_path="best", datamodule=qm9Bygger2)[0]
        for log_metric in self.log_metrics:
            self.resultater[f'{udgave}_{log_metric}'].append(resultater[log_metric])

    def save(self):
        df = pd.DataFrame(data=self.resultater)
        path = os.path.join(self.save_dir, "logs_metrics.csv")
        df.to_csv(path, index=False)

    def main(self):
        self.fortræn()
        for i in range(self.eksp2['trin']):
            eftertræningsandel = self.eftertræningsandele[i].item()
            self.resultater['datamængde'].append(self.get_datamængde(eftertræningsandel))
            for udgave in self.udgaver:
                self.eftertræn(eftertræningsandel=eftertræningsandel, udgave=udgave)
                time.sleep(1)
            self.save()

    def get_datamængde(self, eftertræningsandel):
        n = len(self.qm9Bygger2Hoved.data_splits[False]['train'])
        return int(n*eftertræningsandel)


if __name__ == "__main__":
    args = parserargs()
    eksp2 = Eksp2(args)
    eksp2.main()
