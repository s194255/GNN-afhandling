import lightning as L

import src.redskaber as r
import src.models as m

model = m.Selvvejledt.load_from_checkpoint("artifacts/model-4ci6vjyd-v0/model.ckpt")
print(model.args_dict)