import lightning as L
import src.models as m
import argparse

def med_selvtræn():
    selvvejledt = m.GrundSelvvejledt(debug=args.debug, eftertræningsandel=0.0025,
                                   rygrad_args=m.load_config("config/rygrad_args.yaml"),
                                     hoved_args=m.load_config("config/selvvejledt_hoved_args.yaml"))
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_selvtræn,
                        callbacks=[checkpoint_callback])
    trainer.fit(model=selvvejledt)

    downstream = m.GrundDownstream(debug=args.debug, rygrad_args=m.load_config("config/rygrad_args2.yaml"))
    downstream.indæs_selvvejledt_rygrad(m.GrundSelvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path))
    if args.frys_rygrad:
        downstream.frys_rygrad()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_efterfølgende,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def uden_selvtræn():
    downstream = m.GrundDownstream(debug=args.debug, rygrad_args=m.load_config("config/rygrad_args.yaml"))
    if args.frys_rygrad:
        downstream.frys_rygrad()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_efterfølgende,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--debug', action='store_true', help='Aktiver debug-tilstand')
    parser.add_argument('--epoker_selvtræn', type=int, default=50, help='Antal epoker til selvtræning')
    parser.add_argument('--epoker_efterfølgende', type=int, default=5, help='Antal epoker til efterfølgende træning')
    parser.add_argument('--frys_rygrad', action='store_true', help='Aktiver frossen rygrad-tilstand')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parserargs()
    # DEBUG = args.debug
    # EPOKER_SELVTRÆN = args.epoker_selvtræn
    # EPOKER_EFTERFØLGENDE = args.epoker_efterfølgende
    # FRYS_RYGRAD = args.frys_rygrad
    # RYGRAD_ARGS = m.load_config("config/rygrad_args.yaml")
    # SELVVEJLEDT_HOVED_ARGS = m.load_config("config/selvvejledt_hoved_args.yaml")
    # with open("config/rygrad_args.yaml") as f:
    #     RYGRAD_ARGS = yaml.safe_load(f)
    # with open("config/selvvejledt_hoved_args.yaml") as f:
    #     SELVVEJLEDT_HOVED_ARGS = yaml.safe_load(f)


    med_selvtræn()
    uden_selvtræn()