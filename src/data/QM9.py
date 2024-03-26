import os
import torch

import torch_geometric
import random
from torch_geometric.loader import DataLoader


class QM9Bygger:
    data_splits_path = "data/QM9/processed/data_splits.pt"
    def __init__(self, delmængdestørrelse: float = 0.1, fordeling = None, batch_size=32, num_workers=0):
        if not fordeling:
            fordeling = [0.85, 0.0125, 0.085, 0.0125, 0.04]
        self.root = "data/QM9"
        self.fordeling = torch.tensor(fordeling)
        assert self.fordeling.shape == torch.Size([5])
        assert self.fordeling.sum() == 1
        self.delmængdestørrelse = delmængdestørrelse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_splits_path = QM9Bygger.data_splits_path
        self.mean_std_path = "data/QM9/processed/mean_std.pt"
        self.init_mother_dataset()
        self.init_data_splits()

    def init_mother_dataset(self):
        self.mother_dataset = torch_geometric.datasets.QM9(self.root)
        if os.path.exists(self.mean_std_path):
            self.mean_std = torch.load(self.mean_std_path)
        else:
            self.mean_std = {'means': torch.tensor(self.mother_dataset.mean(0)),
                             'stds': torch.tensor(self.mother_dataset.std(0))}
            torch.save(self.mean_std, self.mean_std_path)
        n = len(self.mother_dataset)
        subset_indices = random.sample(list(range(n)), k=int(self.delmængdestørrelse * n))
        self.mother_dataset = torch.utils.data.Subset(self.mother_dataset, subset_indices)
    def init_data_splits(self):
        if os.path.exists(self.data_splits_path):
            self.data_splits = torch.load(self.data_splits_path)
        else:
            a = torch.multinomial(self.fordeling, len(self.mother_dataset), replacement=True)
            tasks = ['pretrain', 'preval', 'train', 'val', 'test']
            self.data_splits = {}
            for i in range(len(tasks)):
                task = tasks[i]
                self.data_splits[task] = torch.where(a == i)[0]
            torch.save(self.data_splits, self.data_splits_path)
    def get_dataloader(self, dataset, debug, task):
        shuffle_options = {'pretrain': True, 'train': True, 'val': False, 'test': False}
        if debug:
            subset_indices = random.sample(list(range(len(dataset))), k=min(30, len(dataset)))
            dataset = torch.utils.data.Subset(dataset, subset_indices)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=shuffle_options[task], num_workers=self.num_workers)
        return dataloader

    def __call__(self, task: str, debug: bool):
        task_dataset = torch.utils.data.Subset(self.mother_dataset, self.data_splits[task])
        dataloader = self.get_dataloader(task_dataset, debug, task)
        return dataloader
    @staticmethod
    def reset():
        if os.path.exists(QM9Bygger.data_splits_path):
            os.remove(QM9Bygger.data_splits_path)

class QM9Bygger2(QM9Bygger):
    def __init__(self, delmængdestørrelse: float = 0.1):
        self.root = "data/QM9"
        self.fordeling = torch.tensor([0.9, 0.05, 0.05])
        self.delmængdestørrelse = delmængdestørrelse
        self.init_mother_dataset()

        self.init_data_splits()

    def init_data_splits(self):
        a = torch.multinomial(self.fordeling, len(self.mother_dataset), replacement=True)
        tasks = ['pretrain', 'val', 'test']
        self.data_splits = {}
        for i in range(3):
            task = tasks[i]
            self.data_splits[task] = torch.where(a == i)[0]

    def __call__(self, task: str, debug: bool, eftertræningsandel: float = None):
        if task == 'train':
            if not eftertræningsandel:
                raise NotImplementedError
            dataset = torch.utils.data.Subset(self.mother_dataset, self.data_splits['pretrain'])
            subset_indices = random.sample(list(range(len(dataset))),
                                           k=int(eftertræningsandel * len(dataset)))
            dataset = torch.utils.data.Subset(dataset, subset_indices)
        else:
            dataset = torch.utils.data.Subset(self.mother_dataset, self.data_splits[task])
        dataloader = self.get_dataloader(dataset, debug, task)
        return dataloader



