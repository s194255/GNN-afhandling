import lightning as L
import torch

import src.models.visnet_ny.selvvejldt
from src.models import visnet_ny
from src.data import QM9Bygger2




# with torch.no_grad():
model = visnet_ny.VisNetSelvvejledt(debug=True)
# model = visnet_ny.VisNetSelvvejledt2(debug=True)
# model.eval()
# model = src.models.visnet_ny.visnet_ny_skal.ViSNet(derivative=True)
# model = visnet_ny.VisNetDownstream()
# criterion = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.000)
# dataloader = QM9Bygger2()('pretrain', True)
# for data in dataloader:
#     pred, target = model(data.z, data.pos, data.batch)
#     loss = criterion(pred, target)
#     loss.backward()
#     # for name, param in model.named_parameters():
#     #     if param.grad is not None:
#     #         print(f'Parameter: {name}, Gradient shape: {param.grad.shape}, Number of elements: {param.grad.numel()}, grad: {param.grad}')
#     optimizer.step()
#     optimizer.zero_grad()
#     print(loss.item())

checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True)
trainer = L.Trainer(max_epochs=4,
                    inference_mode=False,
                    callbacks=[checkpoint_callback],
                    # track_grad_norm=2,
                    # detect_anomaly=True
                    gradient_clip_val=0.00000001,
                    )
trainer.fit(model)
trainer.test(model=model, ckpt_path='best')