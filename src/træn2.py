print("importerer ...")
import lightning as L
from src.models import visnet_ny
import torch
print("færdig ...")

# model = visnet_ny.VisNetSelvvejledt(debug=True)
# model = visnet_ny.VisNetSelvvejledt2(debug=True)
model = visnet_ny.VisNetDownstream(debug=True, reduce_op='sum', eftertræningsandel=0.20)


checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
trainer = L.Trainer(max_epochs=4,
                    inference_mode=False,
                    callbacks=[checkpoint_callback],
                    # track_grad_norm=2,
                    # detect_anomaly=True
                    gradient_clip_val=0.1,
                    )
trainer.fit(model)
# trainer.test(model=model, ckpt_path='best')