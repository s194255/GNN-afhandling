import argparse
import redskaber as r

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--config', type=str, default="config/fortræn.yaml", help='Sti til eksp2 YAML fil')
    parser.add_argument('--selv_ckpt_path', type=str, default=None, help='Sti til eksp2 YAML fil')
    args = parser.parse_args()
    return args

def main():
    args = parserargs()
    config = r.load_config(args.config)

    selvvejledt, qm9bygger, artefakt_sti = r.get_selvvejledt(config, args.selv_ckpt_path)
    logger = r.wandbLogger(log_model=True, tags=['selvvejledt'])
    trænede_epoker = r.get_n_epoker(artefakt_sti)
    trainer = r.get_trainer(config['selvvejledt']['epoker']+trænede_epoker, logger=logger)
    trainer.fit(model=selvvejledt, datamodule=qm9bygger, ckpt_path=artefakt_sti)

if __name__ == "__main__":
    main()