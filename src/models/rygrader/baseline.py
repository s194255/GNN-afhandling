import lightning as L
import torch

class BaselineRygrad(L.LightningModule):

    def __init__(self,
                 max_z: int = 100,
                 hidden_channels: int = 128
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.motor = torch.nn.Embedding(max_z, hidden_channels)

    def forward(self, z, pos, batch):
        x = self.motor(z)
        v = pos.unsqueeze(-1).expand(-1, -1, self.hidden_channels)
        return x, v, None, None