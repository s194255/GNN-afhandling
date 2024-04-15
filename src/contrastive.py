import redskaber as r
import src.models as m
import src.data as d
from torch_geometric.data.lightning import LightningDataset
import torch_geometric

class QM9ByggerContrastive(d.QM9Bygger):
    def get_mother_dataset(self) -> torch_geometric.data.Dataset:
        return d.QM9Contrastive(self.root)

trainer = r.get_trainer(epoker=10)
eksp2 = m.load_config("config/eksp2_debug.yaml")
eksp2['datasæt']['debug'] = False
model = m.SelvvejledtContrastive(rygrad_args=eksp2['rygrad'],
                                args_dict=eksp2['selvvejledt']['model'])
datamodule = QM9ByggerContrastive(**eksp2['datasæt'], fordeling=[0.2]*5)
trainer.fit(model, datamodule=datamodule)
