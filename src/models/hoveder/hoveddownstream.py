import lightning as L
import torch.nn
from torch_geometric.utils import scatter


class HovedDownstream(L.LightningModule):
    # defaults_args = {'hidden_channels': 128,
    #                  'out_channels': 19,
    #                  'reduce_op': "sum"}

    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 reduce_op: str,
                 ):
        super().__init__()
        self.motor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.reduce_op = reduce_op

    def forward(self, z, pos, batch, x, v):
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)
        return x
