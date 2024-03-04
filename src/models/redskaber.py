from typing import Dict, Any

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


class RiemannGaussian(L.LightningModule):

    def __init__(self,
                 *args,
                 num_steps = 10,
                 step_size = 0.01,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.step_size = step_size

    def _riemann_gaussian_potential(self,
                                    x_current: torch.Tensor,
                                    x_cond: torch.Tensor,
                                    noise_scale: float
                                    ):
        y_current = x_current - torch.mean(x_current)
        y_cond = x_cond - torch.mean(x_cond)
        tæller = torch.norm(y_current.T @ y_current - y_cond.T @ y_cond) ** 2
        nævner = 4 * noise_scale ** 2
        return torch.exp(-tæller / nævner)


    def _calculate_gradient(self,
                            x_current: torch.Tensor,
                            x_cond: torch.Tensor,
                            noise_scale: float
                            ):
        x_clone = x_current.clone().detach()  # Opret en kopi af `x`
        x_clone.requires_grad = True
        potential = self._riemann_gaussian_potential(x_clone, x_cond, noise_scale)
        grad = torch.autograd.grad(potential, x_clone, create_graph=False)[0]
        return grad

    def _langevin_update(self,
                         x_current: torch.Tensor,
                         x_cond: torch.Tensor,
                         noise_scale: float,
                         ):
        grad = self._calculate_gradient(x_current, x_cond, noise_scale)
        noise = torch.randn_like(x_current) * noise_scale * torch.sqrt(torch.tensor(2.0 * self.step_size))
        x = x_current + self.step_size * grad + noise
        return x

    def _get_target(self,
                    x_current: torch.Tensor,
                    x_cond: torch.Tensor,
                    noise_scale: float
                    ):
        y_current = x_current - torch.mean(x_current)
        y_cond = x_cond - torch.mean(x_cond)
        target = (y_current @ y_current.T) @ y_current - (y_current @ y_cond.T) @ y_cond
        target = -1/noise_scale**2 * target
        alpha = torch.norm(y_current @ y_current.T) + torch.norm(y_current @ y_cond.T)
        alpha = alpha/2
        return 1/alpha*target
    def forward(self,
                x: torch.Tensor,
                batch: torch.Tensor,
                noise_idxs: torch.Tensor,
                noise_scales: torch.Tensor,
                **kwargs: Any
                ) -> Any:

        x = x.clone()
        target = torch.empty(size=(x.shape[0], 3), dtype=x.dtype, device=self.device)
        xout = torch.empty(x.shape, dtype=x.dtype, device=self.device)
        for idx in torch.unique(batch):
            x_cond = x[batch == idx]
            x_current = x[batch == idx]
            noise_scale = noise_scales[idx]

            for _ in range(self.num_steps):
                x_current = self._langevin_update(x_current, x_cond, noise_scale)

            target[batch == idx] = self._get_target(x_current, x_cond, noise_scale)
            xout[batch == idx] = x_current

        return xout, target
