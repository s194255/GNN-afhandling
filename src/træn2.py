import lightning as L

import src.models.visnet_ny.selvvejldt
from src.models import visnet_ny
from src.data import QM9Bygger2

# model = visnet_ny.selvvejldt.VisNetSelvvejledt()
# model = src.models.visnet_ny.visnet_ny_skal.ViSNet(derivative=True)
model = visnet_ny.VisNetDownstream()
dataloader = QM9Bygger2()('val', True)
for data in dataloader:
    y = model(data.z, data.pos, data.batch)
    print(y)