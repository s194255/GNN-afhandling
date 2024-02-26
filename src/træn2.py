import lightning as L
from src.models import VisNetSelvvejledt, VisNetDownstream
import argparse
import torch
import pickle
def med_selvtræn(eftertræningsandel):
    selvvejledt = VisNetSelvvejledt(debug=args.debug, eftertræningsandel=eftertræningsandel)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_selvtræn,
                        callbacks=[checkpoint_callback])
    trainer.fit(model=selvvejledt)

    downstream = VisNetDownstream(debug=args.debug)
    downstream.indæs_selvvejledt_rygrad(VisNetSelvvejledt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path))
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_efterfølgende,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    resultater = trainer.test(ckpt_path="best")
    return resultater

def uden_selvtræn():
    downstream = VisNetDownstream(debug=args.debug)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=args.epoker_efterfølgende,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    resultater = trainer.test(ckpt_path="best")
    return resultater

def eksperiment(eftertræningsandel: float):
    res_med_selvtræn = med_selvtræn(eftertræningsandel)
    res_uden_selvtræn = uden_selvtræn()
    return res_med_selvtræn[0]['MSE'], res_uden_selvtræn[0]['MSE']

def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--debug', action='store_true', help='Aktiver debug-tilstand')
    parser.add_argument('--epoker_selvtræn', type=int, default=50, help='Antal epoker til selvtræning')
    parser.add_argument('--epoker_efterfølgende', type=int, default=5, help='Antal epoker til efterfølgende træning')
    parser.add_argument('--trin', type=int, default=10, help='Antal epoker til efterfølgende træning')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parserargs()
    eftertræningsandele = torch.linspace(0.0025, 0.8, steps=args.trin)
    resser_med_selvtræn = []
    resser_uden_selvtræn = []
    for i in range(args.trin):
        res_med_selvtræn, res_uden_selvtræn = eksperiment(eftertræningsandel=eftertræningsandele[i].item())
        resser_med_selvtræn.append(res_med_selvtræn)
        resser_uden_selvtræn.append(res_uden_selvtræn)
    with open("reports/figures/res.pickle", "wb") as f:
        pickle.dump({'uden': resser_uden_selvtræn,
                     'med': resser_med_selvtræn}, f)
