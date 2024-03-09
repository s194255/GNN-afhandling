from typing import Optional, Tuple

from src.models.visnet.kerne import ViSNetBlock, VisNetRyggrad
from src.models.grund import GrundDownstream
class VisNetDownstream(GrundDownstream):

    def __init__(self, *args,
                 max_z: int = 100,
                 lmax: int = 1,
                 vecnorm_type: Optional[str] = None,
                 trainable_vecnorm: bool = False,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 hidden_channels: int = 128,
                 num_rbf: int = 32,
                 trainable_rbf: bool = False,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 vertex: bool = False,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.rygrad = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )