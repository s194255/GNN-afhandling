import lightning as L
from src.models import VisNetSelvvejledt, VisNetDownstream

def selvtræn():
    # selvvejledt = VisNetSelvvejledtPL(debug=True)
    selvvejledt = VisNetSelvvejledt(debug=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=1,
                        callbacks=[checkpoint_callback])
    trainer.fit(model=selvvejledt)
    return trainer.checkpoint_callback.best_model_path

def downstream(best_model_path):
    # downstream = VisNetDownstreamPL(best_model_path, debug=True)
    downstream = VisNetDownstream(debug=True, selvvejledt_ckpt=best_model_path)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
    trainer = L.Trainer(max_epochs=1,
                        callbacks=[checkpoint_callback])
    trainer.fit(downstream)
    trainer.test(ckpt_path="best")

# , default_root_dir='lightning_logs/selvtræn'

if __name__ == "__main__":
    DEBUG = True


    best_model_path = selvtræn()
    downstream(best_model_path)