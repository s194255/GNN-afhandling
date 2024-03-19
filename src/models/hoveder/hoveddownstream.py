import lightning as L
import torch.nn
from torch_geometric.utils import scatter


class HovedDownstream(L.LightningModule):
    args = {'out_channels': 19,
            'reduce_op': "sum"}


    def __init__(self,
                 hidden_channels: int,
                 out_channels: int = args['out_channels'],
                 reduce_op: str = args['reduce_op'],
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
