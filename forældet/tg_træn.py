from torch_geometric.datasets import ZINC, TUDataset, QM9
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, global_mean_pool, models
import torch
import torch.nn.functional as F
from tqdm import tqdm

# from src.models import ViSNet
# models.DimeNet



# dataset = ZINC("../ZINC", subset=True)
# dataset = TUDataset("../TDU", name='MUTAG')
dataset = QM9("../QM9")
dataloader = DataLoader(dataset, shuffle=True, batch_size=2048)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN2, self).__init__()
        torch.manual_seed(12345)
        self.message_passing = Sequential('x, edge_index', [
            (GCNConv(dataset.num_node_features, hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x')
        ])
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.message_passing(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN2(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0
    for data in tqdm(dataloader):
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += loss.item()
        # print(loss.item()/512)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset)}')
