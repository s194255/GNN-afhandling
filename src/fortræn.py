import argparse
import redskaber as r
import wandb
import os
import shutil
import torch

torch.set_float32_matmul_precision('medium')

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
        r.debugify_config(config)

    modelklasse_str = config['modelklasse']
    name = config['datasæt']['name']
    args_dict = config[modelklasse_str][name]['model']
    datasæt_dict = config['datasæt']
    tags = ['selvvejledt', 'qm9bygger']
    epoker = config[modelklasse_str][name]['epoker']
    if config['ckpt']:
        selvvejledt, qm9bygger, artefakt_sti, _ = r.get_selvvejledt_fra_wandb(config, config['ckpt'])
        epoker += r.get_n_epoker(artefakt_sti)
    else:
        selvvejledt, qm9bygger = r.build_selvvejledt(args_dict=args_dict, datasæt_dict=datasæt_dict, modelklasse_str=modelklasse_str)
        artefakt_sti = None
    if config['qm9_path']:
        # _, qm9bygger, _, _ = r.get_selvvejledt_fra_wandb(config, config['qm9_path'])
        qm9bygger = r.get_qm9bygger_fra_wandb(config, config['qm9_path'])
        tags.remove('qm9bygger')

    logger_config = {'opgave': 'fortræn'}
    logger = r.wandbLogger(log_model='all', tags=tags, logger_config=logger_config)
    log_every_n_steps = config[modelklasse_str][name]['log_every_n_steps']
    trainer = r.get_trainer(epoker, logger=logger, log_every_n_steps=log_every_n_steps)
    trainer.fit(model=selvvejledt, datamodule=qm9bygger, ckpt_path=artefakt_sti)
    wandb_run_id = wandb.run.id
    wandb.finish()
    shutil.rmtree(os.path.join("afhandling", wandb_run_id))

if __name__ == "__main__":
    main()