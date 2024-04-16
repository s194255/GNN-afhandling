import copy
import os

import lightning as L
import torch

import torch_geometric
import random
from torch_geometric.loader import DataLoader
from typing import List, Tuple
from torch_geometric.data.data import BaseData
from src.redskaber import RiemannGaussian

DATA_SPLITS_PATH = "data/QM9/processed/data_splits.pt"
ROOT = "data/QM9"

class QM9Contrastive(torch_geometric.datasets.QM9):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = 0.5
        self.riemannGaussian = RiemannGaussian()
    def get(self, idx: int) -> BaseData:
        ægte_molekyle = super().get(idx)
        y = []
        if random.uniform(0, 1) < self.p:
            snydeidx = random.choice(self.indices())
            snydemolekyle = super().get(snydeidx)
            snydepos = snydemolekyle.pos
            snydez = snydemolekyle.z
            y.append(0)
        else:
            batch = torch.zeros(size=ægte_molekyle.z.shape, dtype=torch.int64)
            sigma = 0.01*torch.ones(size=batch.shape)
            snydepos, _ = self.riemannGaussian(ægte_molekyle.pos, batch, sigma)
            snydez = copy.deepcopy(ægte_molekyle.z)
            y.append(1)
        z = torch.concat([ægte_molekyle.z, snydez])
        pos = torch.concat([ægte_molekyle.pos, snydepos])
        y = torch.tensor(y)
        # torch_geometric.data.Data()
        return torch_geometric.data.Data(pos=pos, z=z, y=y)

class QM9Bygger(L.LightningDataModule):
    args = {'delmængdestørrelse': 0.1,
            'fordeling': None,
            'batch_size': 32,
            'num_workers': 0,
            'debug': False,
            }


    def __init__(self,
                 delmængdestørrelse: float,
                 batch_size: int,
                 fordeling: List,
                 num_workers: int,
                 debug: bool,
                 ):
        super().__init__()
        if not fordeling:
            fordeling = [0.85, 0.0125, 0.085, 0.0125, 0.04]
        self.root = ROOT
        self.fordeling = torch.tensor(fordeling)
        self.tasks = ['pretrain', 'preval', 'train', 'val', 'test']
        self.debug = debug
        assert self.fordeling.shape == torch.Size([5])
        assert abs(self.fordeling.sum() - 1) < 10**(-5)
        self.delmængdestørrelse = delmængdestørrelse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_splits_path = DATA_SPLITS_PATH
        self.init_mother_indices()
        self.init_data_splits()
        self.save_hyperparameters()

    def init_mother_indices(self):
        mother_dataset = self.get_mother_dataset()
        n = len(mother_dataset)
        self.mother_indices = random.sample(list(range(n)), k=int(self.delmængdestørrelse * n))
    def init_data_splits(self):
        a = torch.multinomial(self.fordeling, len(self.mother_indices), replacement=True)
        self.data_splits = {False: {}, True: {}}
        # self.data_splits_debug = {}
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            self.data_splits[False][task] = torch.where(a == i)[0].tolist()
            self.data_splits[True][task] = random.sample(
                self.data_splits[False][task],
                k=self.get_debug_k(len(self.data_splits[False][task]))
            )

    def get_debug_k(self, n):
        return max(min(30, n), 1)

    def get_mother_dataset(self) -> torch_geometric.data.Dataset:
        return torch_geometric.datasets.QM9(ROOT)
    def check_splits(self):
        for debug_mode in [True, False]:
            for task1 in self.tasks:
                for task2 in self.tasks:
                    if task1 != task2:
                        intersection = set(self.data_splits[debug_mode][task1]).intersection(
                            set(self.data_splits[debug_mode][task2]))
                        assert len(intersection) == 0
    def setup(self, stage: str) -> None:
        self.check_splits()
        mother_dataset = self.get_mother_dataset()
        self.datasets = {}
        self.datasets_debug = {}
        for task in self.get_setup_tasks(stage):
            self.datasets[task] = torch.utils.data.Subset(mother_dataset, self.data_splits[self.debug][task])

    def get_setup_tasks(self, stage: str):
        if stage == 'fit':
            if self.trainer.model.selvvejledt:
                return ['pretrain', 'preval']
            else:
                return ['train', 'val']
        if stage == 'test':
            return ['test']

    def state_dict(self):
        state = {'data_splits': self.data_splits}
        return state

    def load_state_dict(self, state_dict):
        self.data_splits = state_dict['data_splits']

    def get_dataloader(self, task, shuffle):
        dataset = self.datasets[task]
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def train_dataloader(self):
        task = 'pretrain' if self.trainer.model.selvvejledt else 'train'
        return self.get_dataloader(task, True)

    def val_dataloader(self):
        task = 'preval' if self.trainer.model.selvvejledt else 'val'
        return self.get_dataloader(task, False)

    def test_dataloader(self):
        return self.get_dataloader('test', False)

class QM9Bygger2(QM9Bygger):
    def __init__(self, *args, spænd, n_trin, **kwargs):
        fordeling = self.get_fordeling(spænd[1])
        if 'fordeling' in kwargs.keys():
            assert fordeling == kwargs['fordeling']
        else:
            kwargs = {**kwargs, "fordeling": fordeling}
        super().__init__(*args, **kwargs)
        self.eftertræningsmængder = torch.linspace(spænd[0],
                                                   spænd[1],
                                                   steps=n_trin)
        n = len(self.data_splits[False]['train'])
        assert max(self.eftertræningsmængder) <= n
        self.eftertræningsandele = self.eftertræningsmængder/n
        self.sample_train_reduced(0)

    def get_fordeling(self, højre_interval):
        n = len(self.get_mother_dataset())
        assert højre_interval <= n
        test = 0.04
        train = højre_interval / n * 0.8
        val = højre_interval / n * 0.2
        pretrain = (1 - (test + train + val)) * 0.8
        preval = (1 - (test + train + val)) * 0.2
        return [pretrain, preval, train, val, test]

    def get_eftertræningsmængde(self, i):
        return self.eftertræningsmængder[i].item()

    def sample_train_reduced(self, trin):
        self.trin = trin
        for task in ['train', 'val']:
            for debug_mode in [True, False]:
                data_split = self.data_splits[debug_mode][task]
                andel = self.eftertræningsandele[trin]
                k = max(int(andel * len(data_split)), 1)
                self.data_splits[debug_mode][f'{task}_reduced'] = random.sample(data_split, k=k)

    def get_setup_tasks(self, stage: str):
        setup_tasks = super().get_setup_tasks(stage)
        if (stage == 'fit') and not self.trainer.model.selvvejledt:
            return ['train_reduced', 'val_reduced']
        return setup_tasks

    def train_dataloader(self):
        assert len(self.data_splits[False]['train_reduced']) == self.eftertræningsmængder[self.trin]
        task = 'pretrain' if self.trainer.model.selvvejledt else 'train_reduced'
        return self.get_dataloader(task, True)

    def val_dataloader(self):
        task = 'preval' if self.trainer.model.selvvejledt else 'val_reduced'
        return self.get_dataloader(task, False)

    def state_dict(self):
        state = super().state_dict()
        return {**state, "eftertræningsandele": self.eftertræningsandele}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.eftertræningsandele = state_dict['eftertræningsandele']

def get_metadata():
    root = "data/QM9"
    mother_dataset = torch_geometric.datasets.QM9(root)
    stds = mother_dataset.y.std(dim=0)
    means = mother_dataset.y.mean(dim=0)
    return {'means': means, 'stds': stds}

if __name__ == "__main__":
    qm9 = QM9Bygger2(eftertræningsmængde=500, debug=False, batch_size=1, fordeling=[0.5, 0.2, 0.1, 0.1, 0.1], num_workers=0, delmængdestørrelse=1.0)
    a = 3
