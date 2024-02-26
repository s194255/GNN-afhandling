import random
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from src.data.QM9 import byg_QM9

import lightning as L


class Maskemager(L.LightningModule):

    def forward(self, n_knuder: int,
                edge_index: Tensor,
                maskeringsandel: float) -> Dict[str, Tensor]:
        randperm = torch.randperm(n_knuder, device=self.device)
        k = int(maskeringsandel*n_knuder)
        udvalgte_knuder = randperm[:k]
        edge_index2, _, kantmaske = subgraph(udvalgte_knuder, edge_index, return_edge_mask=True)
        idxs = torch.arange(n_knuder, device=self.device)
        knudemaske = torch.isin(idxs, edge_index2)
        return {'knuder': knudemaske, 'kanter': kantmaske}


def get_dataloader(task: str, debug: bool) -> DataLoader:
    shuffle_options = {'pretrain': True, 'train': True, 'val': False, 'test': False}
    dataset = byg_QM9("data/QM9", task)
    batch_size = 128
    num_workers = 23
    if debug:
        # subset_indices = random.sample(list(range(len(dataset))), k=int(0.1*len(dataset)))
        subset_indices = random.sample(list(range(len(dataset))), k=50)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        batch_size = 8
        num_workers = 0
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle_options[task], num_workers=num_workers)
    return dataloader
