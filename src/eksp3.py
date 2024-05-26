# from src.eksp1 import uden_selvtræn
import argparse
import shutil

import wandb
import os

import src.models as m
import lightning as L
import src.data as d
import src.redskaber
import src.redskaber as r
from lightning.pytorch.loggers import WandbLogger

N = 130831

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 4
    config['datasæt']['num_workers'] = 0
    for opgave in r.get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 5
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8


class Eksp3:

    def __init__(self, args):
        self.args = args
        self.config = src.redskaber.load_config(args.eksp3_path)
        if args.debug:
            debugify_config(self.config)

    def main(self):
        qm9 = d.QM9ByggerEksp3(**self.config['datasæt'])
        metadata = qm9.get_metadata('train')
        downstream = m.DownstreamQM9(
            args_dict=self.config['Downstream']['variant1']['model'],
            metadata=metadata
        )

        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        tags = [downstream.__class__.__name__]
        logger = WandbLogger(project='afhandling', log_model=True, tags=tags,
                             group=f"eksp3")
        trainer = L.Trainer(max_epochs=self.config['Downstream']['variant1']['epoker'],
                            callbacks=callbacks,
                            log_every_n_steps=10,
                            logger=logger)
        trainer.fit(model=downstream, datamodule=qm9)
        trainer.test(ckpt_path="best", datamodule=qm9)
        wandb_run_id = wandb.run.id
        wandb.finish()
        shutil.rmtree(os.path.join("afhandling", wandb_run_id))
def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp3_path', type=str, default="config/eksp3.yaml", help='Sti til eksp1 YAML fil')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Sti til downstream hoved arguments YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parserargs()
    eksp3 = Eksp3(args)
    eksp3.main()