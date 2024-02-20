import lightning as L
from src.models import VisNetSelvvejledt, VisNetDownstream
import argparse

model = VisNetDownstream.load_from_checkpoint("lightning_logs/version_3/checkpoints/best.ckpt")
trainer = L.Trainer()
trainer.test(model=model)