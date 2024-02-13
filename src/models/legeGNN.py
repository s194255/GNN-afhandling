from src.data.legedata import kuldioxid, MolGrafData2
import torch

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        beskednetværk1 = torch.nn.Sequential(torch.nn.Linear(3, 10),
                                             torch.nn.Tanh())
        beskednetværk2 = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                             torch.nn.Tanh())
        self.beskednetværk = [beskednetværk1, beskednetværk2]
        self.udlæsning = torch.nn.Linear(10, 2)

    def forward(self, x, adj):

        for i in range(2):
            besked = self.beskednetværk[i](x)
            # x = adj @ besked
            x = torch.bmm(adj, besked)

        x = x.sum(dim=1)
        out = self.udlæsning(x)
        return out


if __name__ == "__main__":
    gnn = GNN()
    # gnn(kuldioxid)



