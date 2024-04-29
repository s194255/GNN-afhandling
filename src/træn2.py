# print("importerer ...")
# print("færdig ...")
#
# # model = visnet.yaml.VisNetSelvvejledt(debug=True)
# # model = visnet.yaml.VisNetSelvvejledt2(debug=True)
# model = visnet.yaml.VisNetDownstream(debug=True, reduce_op='sum', eftertræningsandel=0.20)
# model.frys_rygrad()
#
#
# checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='loss', mode='min',
#                                                               save_top_k=1, filename='best', save_last=True)
# trainer = L.Trainer(max_epochs=4,
#                     inference_mode=False,
#                     callbacks=[checkpoint_callback],
#                     # track_grad_norm=2,
#                     # detect_anomaly=True
#                     gradient_clip_val=0.1,
#                     )
# trainer.fit(model)
# # trainer.test(model=model, ckpt_path='best')

# model = torch.nn.Linear(1, 1, bias=False)
# rotation_radians = torch.tensor(0.5236)
# rotation_matrix = torch.tensor([
#     [1, 0, 0],
#     [0, torch.cos(rotation_radians), -torch.sin(rotation_radians)],
#     [0, torch.sin(rotation_radians), torch.cos(rotation_radians)]
# ], dtype=torch.float32)
#
#
# a = torch.randn(size=(2, 3, 1))
# b = rotation_matrix @ model(a)
# c = model(rotation_matrix @ a)
# print(b)
# print(c)

import torch_geometric
from torch_geometric.loader import DataLoader

dataset = torch_geometric.datasets.MD17("data/MD17", "malonaldehyde")
dataloader = DataLoader(dataset=dataset, batch_size=2)
for data in dataloader:
    a = 2



