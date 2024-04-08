import os

import lightning as L
import torch

import torch_geometric
import random
from torch_geometric.loader import DataLoader

DATA_SPLITS_PATH = "data/QM9/processed/data_splits.pt"


class QM9Bygger(L.LightningDataModule):
    args = {'delmængdestørrelse': 0.1,
            'fordeling': None,
            'batch_size': 32,
            'num_workers': 0,
            'debug': False,
            }


    def __init__(self,
                 delmængdestørrelse: float = args['delmængdestørrelse'],
                 fordeling=args['fordeling'],
                 batch_size=args['batch_size'],
                 num_workers=args['num_workers'],
                 debug=args['debug']
                 ):
        super().__init__()
        if not fordeling:
            fordeling = [0.85, 0.0125, 0.085, 0.0125, 0.04]
        self.root = "data/QM9"
        self.fordeling = torch.tensor(fordeling)
        self.tasks = ['pretrain', 'preval', 'train', 'val', 'test']
        self.debug = debug
        assert self.fordeling.shape == torch.Size([5])
        assert abs(self.fordeling.sum() - 1) < 10**(-5)
        self.delmængdestørrelse = delmængdestørrelse
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_splits_path = DATA_SPLITS_PATH
        self.init_mother_dataset()
        self.init_data_splits()
        self.save_hyperparameters()

    def init_mother_dataset(self):
        mother_dataset = torch_geometric.datasets.QM9(self.root)
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
        mother_dataset = torch_geometric.datasets.QM9(self.root)
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
    _qm92_args = {'eftertræningsandel': 1.0}
    def __init__(self, *args, eftertræningsandel, **kwargs):
        super().__init__(*args, **kwargs)
        self.eftertræningsandel = eftertræningsandel
        self.sample_train_reduced()

    def sample_train_reduced(self):
        for task in ['train', 'val']:
            for debug_mode in [True, False]:
                data_split = self.data_splits[debug_mode][task]
                k = max(int(self.eftertræningsandel * len(data_split)), 1)
                self.data_splits[debug_mode][f'{task}_reduced'] = random.sample(data_split, k=k)

    def get_setup_tasks(self, stage: str):
        setup_tasks = super().get_setup_tasks(stage)
        if (stage == 'fit') and not self.trainer.model.selvvejledt:
            return ['train_reduced', 'val_reduced']
        return setup_tasks

    def train_dataloader(self):
        assert len(self.data_splits[False]['train_reduced']) == int(self.eftertræningsandel * len(self.data_splits[False]['train']))
        task = 'pretrain' if self.trainer.model.selvvejledt else 'train_reduced'
        return self.get_dataloader(task, True)

    def val_dataloader(self):
        task = 'preval' if self.trainer.model.selvvejledt else 'val_reduced'
        return self.get_dataloader(task, False)

def get_metadata():
    mean_std_path = "data/QM9/processed/mean_std.pt"
    root = "data/QM9"
    if os.path.exists(mean_std_path):
        metadata = torch.load(mean_std_path)
    else:
        mother_dataset = torch_geometric.datasets.QM9(root)
        metadata = {'means': torch.tensor(mother_dataset.mean(0)),
                         'stds': torch.tensor(mother_dataset.std(0))}
        torch.save(metadata, mean_std_path)
    return metadata

if __name__ == "__main__":
    qm9 = QM9Bygger()
    qm9.state_dict()
