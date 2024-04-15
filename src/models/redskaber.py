import random
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.utils import subgraph

import lightning as L

import yaml

from src.redskaber import RiemannGaussian


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


def load_config(path, reference_dict=None):
    with open(path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    if reference_dict:
        config_dict = {key: value for (key, value) in config_dict.items() if key in reference_dict.keys()}
    return config_dict

if __name__ == "__main__":
    riemannGuassian = RiemannGaussian()
    pos = torch.randn((370, 3)) * 1 + 4
    batch = torch.randint(0, 8, (370,))
    sigmas_options = [0.01, 0.1, 1.0, 10.0, 100.0]
    sigma = torch.empty(size=(370,), dtype=torch.float32)
    for i in range(8):
        sigma_ = random.choice(sigmas_options)
        sigma[batch == i] = sigma_

    print(riemannGuassian(pos, batch, sigma))




