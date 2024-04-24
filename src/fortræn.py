import argparse
import redskaber as r
import wandb
import os
import shutil

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--config', type=str, default="config/fortræn.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    parser.add_argument('--debug', action='store_true', help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

def manip_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    # config['downstream']['epoker'] = 1
    config['selvvejledt']['epoker'] = 1
    # config['rygrad']['hidden_channels'] = 8
    # config['selvvejledt']['model']['lr'] = 10**(-3)

def main():
    args = parserargs()
    config = r.load_config(args.config)
    if args.debug:
        manip_config(config)

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