import lightning as L
import torch.nn
from torch_geometric.utils import scatter
from src.models.visnet import GatedEquivariantBlock
from rdkit import Chem


class HovedDownstream(L.LightningModule):
    args = {'out_channels': 1,
            'reduce_op': "sum",
            'num_layers': 2
            }


    def __init__(self,
                 max_z: int,
                 hidden_channels: int,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 num_layers: int = args['num_layers'],
                 out_channels: int = args['out_channels'],
                 reduce_op: str = args['reduce_op'],
                 ):
        super().__init__()
        atom_weights = self.get_atom_weights(max_z)
        self.register_buffer('atom_weights', atom_weights)
        self.motor = torch.nn.ModuleList([])
        for i in range(num_layers-1):
            self.motor.append(GatedEquivariantBlock(hidden_channels,
                                  hidden_channels,
                                  scalar_activation=True))
        self.motor.append(GatedEquivariantBlock(hidden_channels,
                                  out_channels,
                                  scalar_activation=False))
        self.reduce_op = reduce_op
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

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
