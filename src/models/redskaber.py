from typing import Dict

import torch
from torch import Tensor
from torch_geometric.utils import subgraph

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