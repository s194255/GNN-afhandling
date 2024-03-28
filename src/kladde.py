import lightning as L
import src.models as m
from src.models.grund import Selvvejledt2, Downstream
from src.data import QM9Bygger

print(Downstream.grund_args)

# QM9Bygger.reset()
# model = Selvvejledt2(rygrad_args=m.load_config('config/rygrad_args2.yaml'),
#                      **m.load_config('config/debug.yaml', Selvvejledt2.grund_args))
# trainer = L.Trainer(max_epochs=1)
# trainer.fit(model)


