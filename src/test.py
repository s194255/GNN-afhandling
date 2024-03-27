import lightning as L
import src.models as m
from src.models.grund import Selvvejledt2

model = Selvvejledt2(rygrad_args=m.load_config('config/rygrad_args2.yaml'))
trainer = L.Trainer(max_epochs=1)
trainer.fit(model)


# from src.models import VisNetSelvvejledt, VisNetDownstream
import argparse

# model = VisNetDownstream.load_from_checkpoint("lightning_logs/version_3/checkpoints/best.ckpt")
# trainer = L.Trainer()
# trainer.test(model=model)