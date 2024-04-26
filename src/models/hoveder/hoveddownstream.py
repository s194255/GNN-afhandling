import lightning as L
import torch.nn
from torch_geometric.utils import scatter
from src.models.visnet import GatedEquivariantBlock
from rdkit import Chem


class PredictRegular(L.LightningModule):
    args = {
        'reduce_op': "sum",
        'num_layers': 2
    }

    def __init__(self,
                 hidden_channels: int,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 num_layers: int = args['num_layers'],
                 reduce_op: str = args['reduce_op'],
                 ):
        super().__init__()
        self.motor = torch.nn.ModuleList([])
        for i in range(num_layers - 1):
            self.motor.append(GatedEquivariantBlock(hidden_channels,
                                                    hidden_channels,
                                                    scalar_activation=True))
        self.motor.append(GatedEquivariantBlock(hidden_channels,
                                                1,
                                                scalar_activation=False))
        self.reduce_op = reduce_op
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, z, pos, batch, x, v):
        for layer in self.motor:
            x, v = layer(x, v)
        x = x * self.stds + self.means
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return x.squeeze(1)

class PredictDipole(PredictRegular):

    def __init__(self, *args, max_z: int, **kwargs):
        super().__init__(*args, **kwargs)
        atom_weights = self.get_atom_weights(max_z)
        self.register_buffer('atom_weights', atom_weights)

    def get_atom_weights(self, max_z):
        atom_weights = []
        for i in range(max_z):
            atom = Chem.Atom(i)
            mass = atom.GetMass()
            atom_weights.append(mass)
        return torch.tensor(atom_weights)

    def get_centre_of_mass(self, pos, z):
        weights = torch.gather(self.atom_weights, 0, z)
        total_mass = weights.sum()
        weighted_pos = weights.unsqueeze(1) * pos
        return weighted_pos.sum(dim=0)/total_mass

    def forward(self, z, pos, batch, x, v):
        for layer in self.motor:
            x, v = layer(x, v)
        x = x*self.stds + self.means
        r_c = self.get_centre_of_mass(pos, z)
        diff = pos - r_c
        res = x*diff+v.squeeze(2)
        res = scatter(res, batch, dim=0, reduce=self.reduce_op)
        res = torch.linalg.vector_norm(res, dim=1)
        return res

class HovedDownstreamKlogt(L.LightningModule):
    def __init__(self,
                 hidden_channels: int,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 target_idx: int,
                 num_layers: int = 2,
                 reduce_op: str = "sum",
                 max_z: int = 100,
                 ):
        super().__init__()
        if target_idx == 0:
            self.motor = PredictDipole(
                hidden_channels=hidden_channels,
                means=means,
                stds=stds,
                num_layers=num_layers,
                reduce_op=reduce_op,
                max_z=max_z
            )
        else:
            self.motor = PredictRegular(
                hidden_channels=hidden_channels,
                means=means,
                stds=stds,
                num_layers=num_layers,
                reduce_op=reduce_op
            )

    def forward(self, z, pos, batch, x, v):
        return self.motor(z, pos, batch, x, v)

class HovedDownstreamDumt(L.LightningModule):
    def __init__(self,
                hidden_channels: int,
                means: torch.Tensor,
                stds: torch.Tensor,
                num_layers: int = 2,
                reduce_op: str = "sum",
                ):
        super().__init__()
        module_list = []
        for i in range(num_layers-1):
            module_list.append(torch.nn.Linear(hidden_channels, hidden_channels))
            module_list.append(torch.nn.ReLU())
        module_list.append(torch.nn.Linear(hidden_channels, 1))
        self.motor = torch.nn.Sequential(*module_list)
        self.reduce_op = reduce_op
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, z, pos, batch, x, v):
        x = scatter(x, batch, dim=0, reduce=self.reduce_op)
        x = self.motor(x)
        x = x * self.stds + self.means
        return x.squeeze(1)

