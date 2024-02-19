import lightning as L
from src.models import VisNetSelvvejledtPL, VisNetBase, VisNetSelvvejledt

visnetbase = VisNetBase()
vissetselvvejledt = VisNetSelvvejledt(visnetbase)
pl_model = VisNetSelvvejledtPL(vissetselvvejledt, debug=False)
trainer = L.Trainer(max_epochs=1)
trainer.fit(model=pl_model)

