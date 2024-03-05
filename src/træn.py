import lightning as L
from src.models.visnet_ny import VisNetSelvvejledt, VisNetDownstream
import argparse

def med_selvtræn():
    selvvejledt = VisNetSelvvejledt(debug=DEBUG, eftertræningsandel=0.0025)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=EPOKER_SELVTRÆN,
                        callbacks=[checkpoint_callback])
    trainer.fit(model=selvvejledt)

    # downstream = VisNetDownstream(debug=DEBUG, selvvejledt_ckpt=trainer.checkpoint_callback.best_model_path)
    downstream = VisNetDownstream(debug=DEBUG)
    downstream.indæs_selvvejledt_rygrad(VisNetSelvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path))
    if FRYS_RYGRAD:
        downstream.frys_rygrad()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=EPOKER_EFTERFØLGENDE,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

def uden_selvtræn():
    downstream = VisNetDownstream(debug=DEBUG)
    if FRYS_RYGRAD:
        downstream.frys_rygrad()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=EPOKER_EFTERFØLGENDE,
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
    DEBUG = args.debug
    EPOKER_SELVTRÆN = args.epoker_selvtræn
    EPOKER_EFTERFØLGENDE = args.epoker_efterfølgende
    FRYS_RYGRAD = args.frys_rygrad


    med_selvtræn()
    uden_selvtræn()