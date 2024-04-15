import src.models as m
import argparse
import lightning as L

import src.models.downstream


def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--ckpt_path', type=str,
                        help='Sti til downstream hoved arguments YAML fil')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parserargs()
    model = src.models.downstream.Downstream.load_from_checkpoint(args.ckpt_path)
    trainer = L.Trainer()
    trainer.test(model)