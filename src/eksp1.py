import time

import lightning as L
import src.models as m
import src.data as d
import src.redskaber
import src.redskaber as r
import argparse

import src.models.downstream
import src.models.selvvejledt
from src.redskaber import TQDMProgressBar, checkpoint_callback

class Eskp1:

    def __init__(self, args):
        self.config = src.redskaber.load_config(args.config_path)
        self.udgaver = ['med', 'uden']
        self.qm9Bygger = d.QM9ByggerEksp1(**self.config['datasæt'])

    def fortræn(self):
        selvvejledt = src.models.selvvejledt.Selvvejledt(rygrad_args=self.config['rygrad'],
                                                         hoved_args=self.config['selvvejledt']['hoved'],
                                                         args_dict=self.config['selvvejledt']['model'])
        trainer = r.get_trainer(epoker=self.config['selvvejledt']['epoker'])
        trainer.fit(selvvejledt, datamodule=self.qm9Bygger)
        self.bedste_selvvejledt = m.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        time.sleep(1)
    def main(self):
        self.fortræn()
        for frys_rygrad in [False, True]:
            for udgave in self.udgaver:
                self.eftertræn(udgave, frys_rygrad)

def med_selvtræn():
    selvvejledt = src.models.selvvejledt.Selvvejledt(rygrad_args=src.redskaber.load_config(args.rygrad_args_path),
                                                     hoved_args=src.redskaber.load_config(args.selvvejledt_hoved_args_path),
                                                     træn_args=src.redskaber.load_config(args.eksp1_path, src.models.selvvejledt.Selvvejledt.udgngs_træn_args))
    trainer = L.Trainer(max_epochs=eksp1['epoker_selvtræn'],
                        callbacks=[checkpoint_callback(),
                                   TQDMProgressBar()])
    trainer.fit(model=selvvejledt)

    downstream = src.models.downstream.Downstream(rygrad_args=src.redskaber.load_config(args.rygrad_args_path),
                                                  hoved_args=src.redskaber.load_config(args.downstream_hoved_args_path),
                                                  træn_args=src.redskaber.load_config(args.eksp1_path, src.models.downstream.Downstream.udgngs_træn_args))
    downstream.indæs_selvvejledt_rygrad(
        src.models.selvvejledt.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path))
    if eksp1['frys_rygrad']:
        downstream.frys_rygrad()
    trainer = L.Trainer(max_epochs=eksp1['epoker_efterfølgende'],
                        callbacks=[checkpoint_callback(),
                                   TQDMProgressBar()])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def uden_selvtræn():
    downstream = src.models.downstream.Downstream(rygrad_args=src.redskaber.load_config(args.rygrad_args_path),
                                                  hoved_args=src.redskaber.load_config(args.downstream_hoved_args_path),
                                                  træn_args=src.redskaber.load_config(args.eksp1_path, src.models.downstream.Downstream.udgngs_træn_args)
                                                  )
    if eksp1['frys_rygrad']:
        downstream.frys_rygrad()
    trainer = L.Trainer(max_epochs=eksp1['epoker_efterfølgende'],
                        callbacks=[checkpoint_callback(),
                                   TQDMProgressBar()])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--rygrad_args_path', type=str, default="config/rygrad_args.yaml",
                        help='Sti til rygrad arguments YAML fil')
    parser.add_argument('--selvvejledt_hoved_args_path', type=str, default="config/selvvejledt_hoved_args.yaml",
                        help='Sti til selvvejledt hoved arguments YAML fil')
    parser.add_argument('--eksp1_path', type=str, default="config/eksp1.yaml", help='Sti til eksp1 YAML fil')
    parser.add_argument('--downstream_hoved_args_path', type=str, default="config/downstream_hoved_args.yaml",
                        help='Sti til downstream hoved arguments YAML fil')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parserargs()
    eksp1 = src.redskaber.load_config(args.eksp1_path)
    eksp1_model = {key: value for (key, value) in eksp1.items() if key in src.models.selvvejledt.Selvvejledt.udgngs_træn_args.keys()}

    med_selvtræn()
    uden_selvtræn()