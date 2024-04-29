import argparse
import time
import subprocess
import os
import yaml
import pandas as pd
import lightning as L
import wandb

# Importer nødvendige klasser og funktioner fra dit projekt
import src.redskaber as r
import torch
import torchmetrics
import src.models as m
from torch_geometric.data import Data


# Parser arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Eftertræning')
    parser.add_argument('--udgave', type=str, required=True, help='Udgave')
    parser.add_argument('--trin', type=int, required=True, help='Eftertræningstrin')
    parser.add_argument('--frys_rygrad', type=bool, required=True, help='Frys rygrad')
    parser.add_argument('--config_path', type=str, required=True, help='Sti til konfigurationsfil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    return parser.parse_args()

def main():
    args = parse_args()
    config = r.load_config(args.config_path)


    downstream = m.Downstream(args_dict=config['downstream']['model'])
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