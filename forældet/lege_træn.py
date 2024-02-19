from src.data.legedata import MolGrafData, MolGrafData2, skrædder_collate
import torch
from src.models.legeGNN import GNN
from torch.utils.data import DataLoader
from torch.optim import Adam

dataset = MolGrafData2()
dataloader = DataLoader(dataset, batch_size=3, collate_fn=skrædder_collate, shuffle=True)
gnn = GNN()
criterion = torch.nn.CrossEntropyLoss()

optimizer = Adam(gnn.parameters(), lr=0.1)
num_epochs = 50

for epoch in range(num_epochs):
    for parti in dataloader:
        optimizer.zero_grad()
        out = gnn(parti['node_features'], parti['adj'], parti['knude_maske'])
        loss = criterion(out, parti['etikette'])
        loss.backward()
        optimizer.step()
        print(parti['etikette'])
        print(out.argmax(dim=1))

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

print('Træning afsluttet!')