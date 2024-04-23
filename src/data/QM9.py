import copy
import os

import lightning as L
import torch

import torch_geometric
import random
from torch_geometric.loader import DataLoader
from typing import List, Tuple
import itertools
from torch_geometric.data.data import BaseData
from src.redskaber import RiemannGaussian

ROOT = "data/QM9"

class QM9Bygger(L.LightningDataModule):

    def __init__(self,
                 delmængdestørrelse: float,
                 batch_size: int,
                 num_workers: int,
                 debug: bool,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.root = ROOT
        self.mother_indices = self.create_mother_indices(delmængdestørrelse)
        self.data_splits = self.create_data_splits()

    def create_mother_indices(self, delmængdestørrelse: float):
        mother_dataset = self.get_mother_dataset()
        n = len(mother_dataset)
        return random.sample(list(range(n)), k=int(delmængdestørrelse * n))

    def get_mother_dataset(self) -> torch_geometric.data.Dataset:
        return torch_geometric.datasets.QM9(self.root)

    def create_data_splits(self):
        assert self.fordeling.shape == torch.Size([len(self.tasks)])
        assert abs(self.fordeling.sum() - 1) < 10 ** (-5)
        a = torch.multinomial(self.fordeling, len(self.mother_indices), replacement=True)
        data_splits = {False: {}, True: {}}
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            data_splits[False][task] = torch.where(a == i)[0].tolist()
            debug_k = torch.tensor(30).clip(min=1, max=len(data_splits[False][task])).item()
            data_splits[True][task] = random.sample(
                data_splits[False][task],
                k=debug_k
            )
        return data_splits

    def setup(self, stage: str) -> None:
        self.check_splits()
        mother_dataset = self.get_mother_dataset()
        self.datasets = {}
        self.datasets_debug = {}
        for task in self.get_setup_tasks(stage):
            self.datasets[task] = torch.utils.data.Subset(mother_dataset, self.data_splits[self.debug][task])

    @property
    def fordeling(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def tasks(self) -> List:
        return ['pretrain', 'preval', 'train', 'val', 'test']

    def get_setup_tasks(self, stage) -> List[str]:
        if stage == 'fit':
            if self.trainer.model.selvvejledt:
                return ['pretrain', 'preval']
            else:
                return ['train', 'val']
        if stage == 'test':
            return ['test']

    def check_splits(self):
        debug_modes = [True, False]
        task_combinations = list(itertools.combinations(self.tasks, 2))
        for debug_mode, (task1, task2) in itertools.product(debug_modes, task_combinations):
            set1 = set(self.data_splits[debug_mode][task1])
            set2 = set(self.data_splits[debug_mode][task2])
            intersection = set1.intersection(set2)
            assert len(intersection) == 0, f"Overlap mellem {task1} og {task2} i debug_mode {debug_mode}"

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

class QM9ByggerEksp1(QM9Bygger):

    def __init__(self,
                 *args,
                 fordeling: List,
                 **kwargs
                 ):
        self.fordeling_cached = torch.tensor(fordeling)
        super().__init__(*args, **kwargs)
    @property
    def fordeling(self) -> torch.Tensor:
        return self.fordeling_cached

class QM9ByggerEksp2(QM9Bygger):

    def __init__(self, *args, spænd: List, n_trin: int, **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.data_splits[False]['train'])
        self.eftertræningsandele = torch.linspace(spænd[0], spænd[1], steps=n_trin) / n
        self.eftertræningsandele = self.eftertræningsandele.clip(min=0, max=1)
        self.sample_train_reduced(0)

    def sample_train_reduced(self, trin):
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
        task = 'pretrain' if self.trainer.model.selvvejledt else 'train_reduced'
        return self.get_dataloader(task, True)

    def val_dataloader(self):
        task = 'preval' if self.trainer.model.selvvejledt else 'val_reduced'
        return self.get_dataloader(task, False)

    @property
    def fordeling(self) -> torch.Tensor:
        if not hasattr(self, "fordeling_cached"):
            spænd = self.hparams.spænd
            n = len(self.mother_indices)
            assert spænd[1] <= n
            test = 0.04
            train = spænd[1] / n
            val = 0.2 / 0.8 * train
            pretrain = (1 - (test + train + val)) * 0.8
            preval = (1 - (test + train + val)) * 0.2
            self.fordeling_cached = torch.tensor([pretrain, preval, train, val, test])
        return self.fordeling_cached

    def get_eftertræningsmængde(self):
        return len(self.data_splits[False]['train_reduced'])

class QM9ByggerEksp3(QM9Bygger):
    def __init__(self, *args, n_train: int, **kwargs):
        self.fordeling_cached = None
        super().__init__(*args, **kwargs)

    @property
    def fordeling(self) -> torch.Tensor:
        if self.fordeling_cached:
            n = len(self.mother_indices)
            n_train = self.hparams.n_train
            assert n_train < n
            train = n_train / n
            val = 0.25 * train
            test = 0.04
            pretrain = (1 - (train + val + test)) * 0.8
            preval = (1 - (train + val + test)) * 0.2
            self.fordeling_cached = torch.tensor([pretrain, preval, train, val, test])
        return self.fordeling_cached




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


def get_metadata():
    root = "data/QM9"
    mother_dataset = torch_geometric.datasets.QM9(root)
    stds = mother_dataset.y.std(dim=0)
    means = mother_dataset.y.mean(dim=0)
    return {'means': means, 'stds': stds}
