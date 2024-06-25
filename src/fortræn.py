import argparse
import redskaber as r
import wandb
import os
import shutil
import torch

from src.redskaber import get_opgaver_in_config

torch.set_float32_matmul_precision('medium')


def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 4
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    for opgave in get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 2
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--config', type=str, default="config/fortræn.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

def main():
    args = parserargs()
    config = r.load_config(args.config)
    if args.debug:
        debugify_config(config)

    modelklasse_str = config['modelklasse']
    name = config['datasæt']['name']
    args_dict = config[modelklasse_str][name]['model']
    datasæt_dict = config['datasæt']
    tags = ['selvvejledt', 'qm9bygger']
    epoker = config[modelklasse_str][name]['epoker']
    logger_config = {'opgave': 'fortræn'}
    if config['ckpt']:
        selvvejledt, qm9bygger, artefakt_sti, id = r.get_selvvejledt_fra_wandb(config, config['ckpt'])
        epoker += r.get_n_epoker(artefakt_sti, id)+1
        logger_config['ckpt_wandb_path'] = config['ckpt']
        resume = 'must'
    else:
        selvvejledt, qm9bygger = r.build_selvvejledt(args_dict=args_dict, datasæt_dict=datasæt_dict, modelklasse_str=modelklasse_str)
        artefakt_sti = None
        id, resume = None, None
    if config['qm9_path']:
        # _, qm9bygger, _, _ = r.get_selvvejledt_fra_wandb(config, config['qm9_path'])
        qm9bygger = r.get_qm9bygger_fra_wandb(config, config['qm9_path'])
        tags.remove('qm9bygger')

    log_model = config[modelklasse_str][name]['log_model']
    logger = r.wandbLogger(log_model=log_model, tags=tags, logger_config=logger_config, id=id, resume=resume)
    log_every_n_steps = config[modelklasse_str][name]['log_every_n_steps']
    trainer = r.get_trainer(epoker, logger=logger, log_every_n_steps=log_every_n_steps)
    trainer.fit(model=selvvejledt, datamodule=qm9bygger, ckpt_path=artefakt_sti)
    wandb_run_id = wandb.run.id
    wandb.finish()
    shutil.rmtree(os.path.join("afhandling", wandb_run_id))

if __name__ == "__main__":
    main()
