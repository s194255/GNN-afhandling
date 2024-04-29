import lightning as L
import torch
from src.models.rygrader.visnet import GatedEquivariantBlock
from torch_geometric.utils import scatter
class GatedEquivariantMotor(L.LightningModule):

    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 num_layers: int,
                 reduce_op: str,
                 ):
        super().__init__()
        self.motor = torch.nn.ModuleList([])
        for i in range(num_layers - 1):
            self.motor.append(GatedEquivariantBlock(hidden_channels,
                                                    hidden_channels,
                                                    scalar_activation=True))
        self.motor.append(GatedEquivariantBlock(hidden_channels,
                                                out_channels,
                                                scalar_activation=False))
        self.reduce_op = reduce_op
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, z, pos, batch, x, v):
        for layer in self.motor:
            x, v = layer(x, v)
        x = x * self.stds + self.means
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return x

    def reset_parameters(self):
        for layer in self.motor:
            layer.reset_parameters()

class LinearMotor(L.LightningModule):
    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 num_layers: int,
                 reduce_op: str,
                 ):
        super().__init__()
        module_list = []
        for i in range(num_layers - 1):
            module_list.append(torch.nn.Linear(hidden_channels, hidden_channels))
            module_list.append(torch.nn.ReLU())
        module_list.append(torch.nn.Linear(hidden_channels, out_channels))
        self.motor = torch.nn.Sequential(*module_list)
        self.reduce_op = reduce_op
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, z, pos, batch, x, v):
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)
        x = x * self.stds + self.means
        return x
