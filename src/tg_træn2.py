from src.models import ViSNet, VisNetSelvvejledt
import torch_geometric
import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

dataset = torch_geometric.datasets.QM9("../QM9")
# dataset.slices = random.sample(dataset.slices, k=100)
# dataloader = torch_geometric.loader.DataLoader(dataset, shuffle=True, batch_size=8)
subset_indices = random.sample(list(range(len(dataset))), k=50)
subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
dataloader = torch_geometric.loader.DataLoader(dataset, sampler=subset_sampler, batch_size=8, shuffle=False)

# model = ViSNet()
model = VisNetSelvvejledt()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss(reduction='mean')
num_epochs = 20
subset_size = 100

total_losses = []
for epoch in range(num_epochs):
    total_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        pred, target = model(data.z, data.pos, data.batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_losses.append(total_loss)
    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset)}')
plt.plot(total_losses)
plt.savefig("reports/figures/a.png")