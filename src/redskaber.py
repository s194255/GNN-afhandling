import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import os

def TQDMProgressBar():
    return L.pytorch.callbacks.TQDMProgressBar(refresh_rate=1000)


def checkpoint_callback():
    return  L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
def tensorBoardLogger(save_dir=None, name=None, version=None):
    if not save_dir:
        save_dir = os.getcwd()
    if not name:
        name = "lightning_logs"
    return TensorBoardLogger(save_dir=save_dir, name=name, version=version)

def get_trainer(epoker, logger=None):
    callbacks = [
        checkpoint_callback(),
        TQDMProgressBar(),
    ]
    trainer = L.Trainer(max_epochs=epoker,
                        log_every_n_steps=1,
                        callbacks=callbacks,
                        logger=logger,
                        )
    return trainer