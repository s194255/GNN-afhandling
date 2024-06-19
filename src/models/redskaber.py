import random
from typing import Dict

import torch
from torch import Tensor
from torch_geometric.utils import subgraph

import lightning as L

import yaml
from torch_scatter import scatter_mean, scatter_add
import math


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


def save_config(config, path):
    with open(path, 'w', encoding='utf-8') as fil:
        yaml.dump(config, fil, allow_unicode=True)


# class RiemannGaussian(L.LightningModule):
#
#     def __init__(self):
#         super().__init__()
#         # TODO: gør så man kan bruge T'er
#         self.T = 1
#
#     @torch.no_grad()
#     def get_s(self, pos_til, pos, batch, sigma):
#         v = pos.shape[-1]
#         center = scatter_mean(pos, batch, dim=-2)  # B * 3
#         perturbed_center = scatter_mean(pos_til, batch, dim=-2)  # B * 3
#         pos_c = pos - center[batch]
#         perturbed_pos_c = pos_til - perturbed_center[batch]
#         perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v, dim=-1)
#         perturbed_pos_c_right = perturbed_pos_c.repeat([1, v])
#         pos_c_left = pos_c.repeat_interleave(v, dim=-1)
#         ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v,
#                                                                                                v)  # B * 3 * 3
#         otp = scatter_add(pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v, v)  # B * 3 * 3
#         ptp = ptp[batch]
#         otp = otp[batch]
#         # s = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (
#         #         torch.norm(ptp, dim=(1, 2)) + torch.norm(otp, dim=(1, 2))).unsqueeze(-1).repeat([1, 3])
#         s = (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1)
#         s = -(1/sigma**2).view(-1, 1) * s
#         alpha = (torch.norm(ptp, dim=(1, 2)) + torch.norm(otp, dim=(1, 2)))/2
#         return s, alpha
#     @torch.no_grad()
#     def forward(self,
#                 pos: torch.Tensor,
#                 batch: torch.Tensor,
#                 sigma: torch.Tensor,
#                 ):
#         pos_til = pos.clone()
#         for t in range(1, self.T+1):
#             beta = (sigma**2)/(2**t)
#             s, alpha = self.get_s(pos_til, pos, batch, sigma)
#             pos_til = pos_til + (beta/alpha).view(-1, 1) * s + torch.sqrt(2*beta).view(-1, 1)*torch.randn_like(pos)
#         target = (1/alpha).view(-1, 1) * s
#         return pos_til, target

class RiemannGaussian(L.LightningModule):

    @torch.no_grad()
    def get_force_target(self, pos_til, pos, batch):
        v = pos.shape[-1]
        center = scatter_mean(pos, batch, dim=-2)  # B * 3
        perturbed_center = scatter_mean(pos_til, batch, dim=-2)  # B * 3
        pos_c = pos - center[batch]
        perturbed_pos_c = pos_til - perturbed_center[batch]
        perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v, dim=-1)
        perturbed_pos_c_right = perturbed_pos_c.repeat([1, v])
        pos_c_left = pos_c.repeat_interleave(v, dim=-1)
        ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v,
                                                                                               v)  # B * 3 * 3
        otp = scatter_add(pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v, v)  # B * 3 * 3
        ptp = ptp[batch]
        otp = otp[batch]
        tar_force = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (
                torch.norm(ptp, dim=(1, 2)) + torch.norm(otp, dim=(1, 2))).unsqueeze(-1).repeat([1, 3])
        return tar_force

    @torch.no_grad()
    def forward(self,
                pos: torch.Tensor,
                batch: torch.Tensor,
                sigma: torch.Tensor,
                ):
        pos_til = pos
        steps = 1
        sigma = sigma.unsqueeze(-1)
        for t in range(1, steps + 1):
            alpha = 1 / (2 ** t)
            s = self.get_force_target(pos_til, pos, batch)
            pos_til = pos_til + alpha * s + torch.randn_like(pos) * math.sqrt(2 * alpha) * sigma
        target = self.get_force_target(pos_til, pos, batch) / sigma
        return pos_til, target

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




