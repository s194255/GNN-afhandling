import argparse
import redskaber as r
import wandb
import os
import shutil
import src.models as m
import src.data as d


class SelvvejledtQM9(m.Downstream):
    @property
    def selvvejledt(self):
        return True

def get_selvvejledtQM9(config, selv_ckpt_path):
    if selv_ckpt_path:
        artefakt_sti, run_id =  r.indlæs_selv_ckpt_path(selv_ckpt_path)
        selvvejledt = SelvvejledtQM9.load_from_checkpoint(artefakt_sti)
        qm9bygger = d.QM9ByggerEksp2.load_from_checkpoint(artefakt_sti, **config['datasæt'])
    else:
        selvvejledt = SelvvejledtQM9(rygrad_args=config['rygrad'],
                                    args_dict=config['downstream']['model'])
        qm9bygger = d.QM9ByggerEksp2(**config['datasæt'])
        artefakt_sti = None
        run_id = None
    return selvvejledt, qm9bygger, artefakt_sti, run_id



def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--config', type=str, default="config/fortræn.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--selvQM9', action='store_true', help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

def manip_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    config['selvvejledt']['epoker'] = 1
    config['rygrad']['hidden_channels'] = 8

def main():
    args = parserargs()
    config = r.load_config(args.config)
    if args.debug:
        r.debugify_config(config)

    if args.selvQM9:
        selvvejledt, qm9bygger, artefakt_sti, _ = get_selvvejledtQM9(config, args.selv_ckpt_path)
    else:
        selvvejledt, qm9bygger, artefakt_sti, _ = r.get_selvvejledt(config, args.selv_ckpt_path)
    logger = r.wandbLogger(log_model=True, tags=['selvvejledt'])
    trænede_epoker = r.get_n_epoker(artefakt_sti)
    trainer = r.get_trainer(config['selvvejledt']['epoker']+trænede_epoker, logger=logger)
    trainer.fit(model=selvvejledt, datamodule=qm9bygger, ckpt_path=artefakt_sti)
    wandb_run_id = wandb.run.id
    wandb.finish()
    shutil.rmtree(os.path.join("afhandling", wandb_run_id))

if __name__ == "__main__":
    main()