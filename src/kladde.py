import lightning as L
import src.models as m
from src.data.QM9 import QM9Bygger
from src.models.grund import Downstream
import torch
from src.redskaber import checkpoint_callback
import time

# print(Downstream.grund_args)

# QM9Bygger.reset()
# model = Selvvejledt2(rygrad_args=m.load_config('config/rygrad_args2.yaml'),
#                      **m.load_config('config/debug.yaml', Selvvejledt2.grund_args))
# trainer = L.Trainer(max_epochs=1)
# trainer.fit(model)

tasks = ['pretrain', 'preval', 'val', 'train', 'test']
debug_modes = [True, False]

model = Downstream(træn_args=m.load_config("config/eksp3_debug.yaml"))
datamodule = QM9Bygger(delmængdestørrelse=0.5)
trainer = L.Trainer(max_epochs=1,
                    callbacks=[checkpoint_callback()])
trainer.fit(model, datamodule=datamodule)

time.sleep(2)
model2 = Downstream.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
datamodule2 = QM9Bygger.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, delmængdestørrelse=0.8)
for debug_mode in debug_modes:
    for task in tasks:
        assert datamodule.data_splits[debug_mode][task] == datamodule2.data_splits[debug_mode][task], f'task {task} fejlede'
        assert len(set(datamodule.data_splits[debug_mode][task])) == len(datamodule.data_splits[debug_mode][task])
for task1 in tasks:
    for task2 in tasks:
        if task1 != task2:
            intersection = set(datamodule.data_splits[debug_mode][task1]).intersection(set(datamodule.data_splits[debug_mode][task2]))
            assert len(intersection) == 0
trainer2 = L.Trainer(max_epochs=1,
                    callbacks=[checkpoint_callback()])
trainer2.fit(model2, datamodule=datamodule2)
for debug_mode in debug_modes:
    for task in tasks:
        assert datamodule.data_splits[debug_mode][task] == datamodule2.data_splits[debug_mode][task], f'task {task} fejlede'
        assert len(set(datamodule.data_splits[debug_mode][task])) == len(datamodule.data_splits[debug_mode][task])
for task1 in tasks:
    for task2 in tasks:
        if task1 != task2:
            intersection = set(datamodule.data_splits[debug_mode][task1]).intersection(set(datamodule.data_splits[debug_mode][task2]))
            assert len(intersection) == 0