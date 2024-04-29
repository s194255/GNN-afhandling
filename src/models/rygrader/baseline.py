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
        n = z.shape[0]
        x = self.motor(z)
        v = torch.zeros(size=(n, 3, self.hidden_channels), dtype=torch.float32, device=self.device)
        return x, v, None, None