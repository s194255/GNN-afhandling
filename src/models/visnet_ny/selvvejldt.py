from typing import Optional
from src.models.visnet_ny.kerne import ViSNetBlock, Distance
from src.models.grund import GrundSelvvejledt


class VisNetSelvvejledt(GrundSelvvejledt):
    def __init__(self, *args,
                 lmax: int = 1,
                 vecnorm_type: Optional[str] = None,
                 trainable_vecnorm: bool = False,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_rbf: int = 32,
                 trainable_rbf: bool = False,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 vertex: bool = False,
                 hidden_channels: int = 128,
                 max_z: int = 100,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.rygrad = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
            hidden_channels=hidden_channels,
            max_z=max_z
        )




