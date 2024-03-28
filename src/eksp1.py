import lightning as L
import src.models as m
import argparse


def checkpoint_callback():
    return  L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
def TQDMProgressBar():
    return L.pytorch.callbacks.TQDMProgressBar(refresh_rate=1000)

def med_selvtræn():
    selvvejledt = m.Selvvejledt(rygrad_args=m.load_config(args.rygrad_args_path),
                                hoved_args=m.load_config(args.selvvejledt_hoved_args_path),
                                træn_args=m.load_config(args.eksp1_path, m.Selvvejledt.udgngs_træn_args))
    trainer = L.Trainer(max_epochs=eksp1['epoker_selvtræn'],
                        callbacks=[checkpoint_callback(),
                                   TQDMProgressBar()])
    trainer.fit(model=selvvejledt)

    downstream = m.Downstream(rygrad_args=m.load_config(args.rygrad_args_path),
                              hoved_args=m.load_config(args.downstream_hoved_args_path),
                              træn_args=m.load_config(args.eksp1_path, m.Downstream.udgngs_træn_args))
    downstream.indæs_selvvejledt_rygrad(m.Selvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path))
    if eksp1['frys_rygrad']:
        downstream.frys_rygrad()
    trainer = L.Trainer(max_epochs=eksp1['epoker_efterfølgende'],
                        callbacks=[checkpoint_callback(),
                                   TQDMProgressBar()])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def uden_selvtræn():
    downstream = m.Downstream(rygrad_args=m.load_config(args.rygrad_args_path),
                              hoved_args=m.load_config(args.downstream_hoved_args_path),
                              træn_args=m.load_config(args.eksp1_path, m.Downstream.udgngs_træn_args)
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
    eksp1 = m.load_config(args.eksp1_path)
    eksp1_model = {key: value for (key, value) in eksp1.items() if key in m.Selvvejledt.udgngs_træn_args.keys()}

    med_selvtræn()
    uden_selvtræn()