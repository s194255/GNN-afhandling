import argparse
import time
import subprocess
import os
import yaml
import pandas as pd
import lightning as L
import wandb

# Importer nødvendige klasser og funktioner fra dit projekt
from src.data import QM9Bygger2
import src.redskaber as r
import torch
import torchmetrics
import src.models as m
from torch_geometric.data import Data

class DownstreamEksp2_2(m.Downstream):
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


# Parser arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Eftertræning')
    parser.add_argument('--trin', type=int, required=True, help='Eftertræningstrin')
    parser.add_argument('--udgave', type=str, required=True, help='Udgave')
    parser.add_argument('--frys_rygrad', type=bool, required=True, help='Frys rygrad')
    parser.add_argument('--config_path', type=str, required=True, help='Sti til konfigurationsfil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    return parser.parse_args()

def main():
    args = parse_args()

    # Indlæs konfiguration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    downstream = DownstreamEksp2_2(config)
    downstream.sample_train_reduced(args.trin)

    if args.udgave == 'med':
        selvvejledt = m.Selvvejledt.load_from_checkpoint()
        downstream.indæs_selvvejledt_rygrad(config['bedste_selvvejledt'])

    if args.frys_rygrad:
        downstream.frys_rygrad()

    # Indstil træner
    trainer = L.Trainer(...)

    # Kør træning
    trainer.fit(model=downstream)

    # Test og få resultater
    resultater = trainer.test()[0]

    # Returner resultater
    print(yaml.dump(resultater))

if __name__ == "__main__":
    main()