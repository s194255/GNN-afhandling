import random
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from src.data.QM9 import byg_QM9
from src.models.tg_kilde import ViSNetBlock, EquivariantScalar, Atomref

import lightning as L


class Maskemager(L.LightningModule):

    def forward(self, n_knuder: int,
                edge_index: Tensor,
                maskeringsandel: float) -> Dict[str, Tensor]:
        randperm = torch.randperm(n_knuder, device=self.device)
        k = int(maskeringsandel*n_knuder)
        udvalgte_knuder = randperm[:k]
        edge_index2, _, kantmaske = subgraph(udvalgte_knuder, edge_index, return_edge_mask=True)
        idxs = torch.arange(n_knuder)
        knudemaske = torch.isin(idxs, edge_index2)
        return {'knuder': knudemaske, 'kanter': kantmaske}


class VisNetBase(torch.nn.Module):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 19,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[Tensor] = None,
        std: float = 1.0,
    ) -> None:
        super().__init__()

        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        self.output_model = EquivariantScalar(hidden_channels=hidden_channels, out_channels=out_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z)
        self.hidden_channels = hidden_channels
        self.register_buffer('std', torch.tensor(std))
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec: Tensor,
        masker: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x, v, edge_attr = self.representation_model(z, pos, batch,
                                                    edge_index, edge_weight, edge_vec, masker)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        return x, edge_attr, edge_index


def get_dataloader(task: str, debug: bool) -> DataLoader:
    shuffle_options = {'pretrain': True, 'train': True, 'val': False, 'test': False}
    dataset = byg_QM9("data/QM9", task)
    if debug:
        subset_indices = random.sample(list(range(len(dataset))), k=50)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=shuffle_options[task], num_workers=0)
    return dataloader
