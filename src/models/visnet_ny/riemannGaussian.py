import torch
import lightning as L
from typing import Any

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
                noise_scale: float = 1.0,
                **kwargs: Any
                ) -> Any:

        x_cond = x.clone()
        x_current = x.clone()

        for _ in range(self.num_steps):
            x_current = self._langevin_update(x_current, x_cond, noise_scale)

        target = self._get_target(x_current, x_cond, noise_scale)

        return x_current, target

if __name__ == "__main__":

    rimannGaussian = RiemannGaussian()
    x_init = torch.tensor([[0.5, 1.0],
                           [0.5, 1.0]])
    print(x_init)
    x_sampled = rimannGaussian(x_init)
    print(x_sampled)