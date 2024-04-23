import redskaber as r
import src.data.QM9
import src.models as m
import src.data as d
from torch_geometric.data.lightning import LightningDataset
import torch_geometric

import src.redskaber


class QM9ByggerContrastive(d.QM9Bygger):
    def get_mother_dataset(self) -> torch_geometric.data.Dataset:
        return src.data.QM9ny.QM9Contrastive(self.root)

trainer = r.get_trainer(epoker=10)
eksp2 = src.redskaber.load_config("config/eksp2_debug.yaml")
eksp2['datasæt']['debug'] = False
model = m.SelvvejledtContrastive(rygrad_args=eksp2['rygrad'],
                                args_dict=eksp2['selvvejledt']['model'])
datamodule = QM9ByggerContrastive(**eksp2['datasæt'])
trainer.fit(model, datamodule=datamodule)
