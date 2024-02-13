import torch
from torch.utils.data import Dataset

class MolGraf:
    def __init__(self, adjecency_matrix, node_features):
        self.adjecency_matrix = adjecency_matrix
        self.node_features = node_features

kuldioxid = MolGraf(torch.tensor([[1, 1, 0],
                                [1, 1, 1],
                                [0, 1, 1]], dtype=torch.float32), torch.tensor([[1, 0, 0],
                                                            [0, 1, 0],
                                                            [1, 0, 0]], dtype=torch.float32))

vand = MolGraf(torch.tensor([[1, 1, 0],
                             [1, 1, 1],
                             [0, 1, 1]], dtype=torch.float32), torch.tensor([[1, 0, 0],
                                                        [0, 0, 1],
                                                        [1, 0, 0]], dtype=torch.float32))


class MolGrafData(Dataset):

    def __init__(self):
        self.molekyler = [kuldioxid, vand]

    def __getitem__(self, idx):
        return self.molekyler[idx]

    def __len__(self):
        return len(self.molekyler)


class MolGrafData2(Dataset):

    def __init__(self):
        kuldioxid = self.get_kuldioxid()
        vand = self.get_vand()
        metan = self.get_metan()
        self.molekyler = [kuldioxid, vand, metan]

    def get_kuldioxid(self):
        adj = torch.tensor([[1, 1, 0],
                            [1, 1, 1],
                            [0, 1, 1]], dtype=torch.float32)
        node_features = torch.tensor([[0, 0, 1],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=torch.float32)
        etikette = torch.tensor(1)
        return {'node_features': node_features, 'adj': adj, 'etikette': etikette}

    def get_vand(self):
        adj = torch.tensor([[1, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 1]], dtype=torch.float32)
        node_features = torch.tensor([[1, 0, 0],
                            [0, 0, 1],
                            [1, 0, 0]], dtype=torch.float32)
        etikette = torch.tensor(0)
        return {'node_features': node_features, 'adj': adj, 'etikette': etikette}

    def get_metan(self):
        adj = torch.tensor([[1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 1],
                            [0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 1],
                            [1, 1, 1, 1, 1]], dtype=torch.float32)

        node_features = torch.tensor([[1, 0, 0],
                                      [1, 0, 0],
                                      [1, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 0]], dtype=torch.float32)
        etikette = torch.tensor(1)
        return {'node_features': node_features, 'adj': adj, 'etikette': etikette}



    def __getitem__(self, item):
        return self.molekyler[item]

    def __len__(self):
        return len(self.molekyler)


def skrædder_collate(parti):
    max_knude_antal = max([graf['node_features'].shape[0] for graf in parti])
    partistørrelse = len(parti)
    knude_dim = parti[0]['node_features'].shape[1]
    node_features = torch.zeros((partistørrelse, max_knude_antal, knude_dim), dtype=torch.float32)
    adjs = torch.zeros((partistørrelse, max_knude_antal, max_knude_antal), dtype=torch.float32)
    etiketter = torch.zeros(partistørrelse, dtype=torch.long)
    knude_masker = torch.zeros((partistørrelse, max_knude_antal))
    for i in range(len(parti)):
        graf = parti[i]
        antal_knuder = graf['node_features'].shape[0]
        node_feature = torch.zeros((max_knude_antal, knude_dim), dtype=torch.float32)
        node_feature[:antal_knuder] = graf['node_features']
        adj = torch.zeros((max_knude_antal, max_knude_antal), dtype=torch.float32)
        adj[:antal_knuder, :antal_knuder] = graf['adj']
        etikette = graf['etikette']
        knude_maske = torch.zeros((max_knude_antal), dtype=torch.bool)
        knude_maske[:antal_knuder] = torch.ones(antal_knuder)

        node_features[i] = node_feature
        adjs[i] = adj
        etiketter[i] = etikette
        knude_masker[i] = knude_maske
    return {'node_features': node_features, 'adj': adjs,
            'etikette': etiketter, 'knude_maske': knude_masker}

