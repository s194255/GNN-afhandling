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
    args_dict = config[modelklasse_str]['variant1']['model']
    datasæt_dict = config['datasæt']
    tags = ['selvvejledt', 'qm9bygger']
    epoker = config[modelklasse_str]['variant1']['epoker']
    if config['ckpt']:
        selvvejledt, qm9bygger, artefakt_sti, _ = r.get_selvvejledt_fra_wandb(config, config['ckpt'])
        epoker += r.get_n_epoker(artefakt_sti)
    else:
        selvvejledt, qm9bygger = r.build_selvvejledt(args_dict=args_dict, datasæt_dict=datasæt_dict, modelklasse_str=modelklasse_str)

    if config['qm9_path']:
        _, qm9bygger, _, _ = r.get_selvvejledt_fra_wandb(config, config['qm9_path'])
        tags.remove('qm9bygger')

    logger_config = {'opgave': 'fortræn'}
    logger = r.wandbLogger(log_model=True, tags=tags, logger_config=logger_config)
    trainer = r.get_trainer(epoker, logger=logger)
    trainer.fit(model=selvvejledt, datamodule=qm9bygger, ckpt_path=config['ckpt'])
    wandb_run_id = wandb.run.id
    wandb.finish()
    shutil.rmtree(os.path.join("afhandling", wandb_run_id))

if __name__ == "__main__":
    main()