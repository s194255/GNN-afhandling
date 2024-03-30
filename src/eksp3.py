# from src.eksp1 import uden_selvtræn
import argparse
import src.models as m
import lightning as L
import src.data as d

def uden_selvtræn():
    if args.ckpt_path:
        print("DU GAV EN CKPT-PATH. Bruger denne og ingorerer visse konfigurationsfiler")
        downstream = m.Downstream.load_from_checkpoint(args.ckpt_path)
        qm9 = d.QM9Byggerlol.load_from_checkpoint(args.ckpt_path)
    else:
        qm9 = d.QM9Byggerlol(**m.load_config(args.eksp3_path, d.QM9Byggerlol.args))
        downstream = m.Downstream(rygrad_args=m.load_config(args.rygrad_args_path),
                                  hoved_args=m.load_config(args.downstream_hoved_args_path),
                                  træn_args=eksp3)
    if eksp3['frys_rygrad']:
        downstream.frys_rygrad()

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=eksp3['epoker_efterfølgende'],
                        callbacks=[checkpoint_callback,
                                   L.pytorch.callbacks.TQDMProgressBar(refresh_rate=1000)
                                   ])
    trainer.fit(downstream, datamodule=qm9, ckpt_path=args.ckpt_path)
    trainer.test(ckpt_path="best", datamodule=qm9)
def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--rygrad_args_path', type=str, default="config/rygrad_args.yaml",
                        help='Sti til rygrad arguments YAML fil')
    parser.add_argument('--selvvejledt_hoved_args_path', type=str, default="config/selvvejledt_hoved_args.yaml",
                        help='Sti til selvvejledt hoved arguments YAML fil')
    parser.add_argument('--eksp3_path', type=str, default="config/eksp3.yaml", help='Sti til eksp1 YAML fil')
    parser.add_argument('--downstream_hoved_args_path', type=str, default="config/downstream_hoved_args.yaml",
                        help='Sti til downstream hoved arguments YAML fil')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Sti til downstream hoved arguments YAML fil')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # QM9Bygger.reset()
    args = parserargs()
    eksp3 = m.load_config(args.eksp3_path)
    uden_selvtræn()