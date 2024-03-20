# from src.eksp1 import uden_selvtræn
import argparse
import src.models as m
import lightning as L
from src.data import QM9Bygger

def uden_selvtræn():
    downstream = m.Downstream(rygrad_args=m.load_config(args.rygrad_args_path),
                              hoved_args=m.load_config(args.downstream_hoved_args_path),
                              **eksp3_model)
    if eksp3['frys_rygrad']:
        downstream.frys_rygrad()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=eksp3['epoker_efterfølgende'],
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")
def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--rygrad_args_path', type=str, default="config/rygrad_args.yaml",
                        help='Sti til rygrad arguments YAML fil')
    parser.add_argument('--selvvejledt_hoved_args_path', type=str, default="config/selvvejledt_hoved_args.yaml",
                        help='Sti til selvvejledt hoved arguments YAML fil')
    parser.add_argument('--eksp3_path', type=str, default="config/eksp3.yaml", help='Sti til eksp1 YAML fil')
    parser.add_argument('--downstream_hoved_args_path', type=str, default="config/downstream_hoved_args.yaml",
                        help='Sti til downstream hoved arguments YAML fil')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    QM9Bygger.reset()
    args = parserargs()
    eksp3 = m.load_config(args.eksp3_path)
    eksp3_model = {key: value for (key, value) in eksp3.items() if key in m.Selvvejledt.træn_args.keys()}
    uden_selvtræn()