import lightning as L
import torch
from torch.nn import Embedding, LayerNorm, Linear, Parameter

class BaselineRygrad(L.LightningModule):

    def __init__(self,
                 max_z: int = 100,
                 hidden_channels: int = 128
                 ):
        super().__init__()
        self.motor = torch.nn.Embedding(max_z, hidden_channels)

    def forward(self, z, pos, batch):
        x = self.embedding(z)
        return x, None, None