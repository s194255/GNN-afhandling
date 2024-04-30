import argparse
import os
import lightning as L
import wandb

import src.redskaber as r
import src.models as m
from lightning.pytorch.loggers import WandbLogger
from src.eksp2 import debugify_config
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Eftertræning')
    parser.add_argument('--udgave', type=str, required=True, help='Udgave (f.eks. med eller uden)')
    parser.add_argument('--trin', type=int, required=True, help='Eftertræningstrin (heltal)')
    parser.add_argument('--temperatur', type=str, required=True, help='Temperatur (f.eks. frossen eller optøet)')
    parser.add_argument('--config_path', type=str, required=True, help='Sti til konfigurationsfil')
    parser.add_argument('--artefakt_sti', type=str, required=True, help='Sti til checkpoint-fil')
    parser.add_argument('--kørselsid', type=int, required=True, help='Kørsels-ID (unik værdi)')
    parser.add_argument('--debug', type=str, required=True, help='Sti til eksp2 YAML fil')
    parser.add_argument('--run_id', type=str, required=True)

    return parser.parse_args()

def get_trainer(config, kørselsid, tags=[]):
    callbacks = [
        r.checkpoint_callback(),
        # r.TQDMProgressBar(),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
    ]
    logger = WandbLogger(project='afhandling', log_model=False, tags=['downstream']+tags,
                         group=f"eksp2_{kørselsid}")
    trainer = L.Trainer(max_epochs=config['downstream']['epoker'],
                        log_every_n_steps=1,
                        callbacks=callbacks,
                        logger=logger,
                        check_val_every_n_epoch=config['downstream']['check_val_every_n_epoch'],
                        enable_progress_bar=False,
                        )
    return trainer

def main():
    args = parse_args()
    config = r.load_config(args.config_path)
    debug = args.debug == 'True'
    if debug:
        debugify_config(config)

    # selvvejledt, qm9Bygger, _, run_id = r.get_selvvejledt_fra_wandb(config, args.selv_ckpt_path)
    selvvejledt, qm9Bygger = r.get_selvvejledt_fra_artefakt_sti(config, args.artefakt_sti)
    config['downstream']['model']['rygrad'] = selvvejledt.args_dict['rygrad']
    downstream = m.Downstream(args_dict=config['downstream']['model'])
    qm9Bygger.sample_train_reduced(args.trin)

    if args.udgave == 'med':
        downstream.indæs_selvvejledt_rygrad(selvvejledt)
    if args.temperatur == "frossen":
        downstream.frys_rygrad()

    tags = [args.udgave, args.temperatur, args.run_id]
    trainer = get_trainer(config, args.kørselsid, tags)

    # Kør træning
    trainer.fit(model=downstream, datamodule=qm9Bygger)
    trainer.test(ckpt_path="best", datamodule=qm9Bygger)
    wandb_run_id = wandb.run.id
    wandb.finish()
    shutil.rmtree(os.path.join("afhandling", wandb_run_id))

if __name__ == "__main__":
    main()