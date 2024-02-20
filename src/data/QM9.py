import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import one_hot, scatter
import torch_geometric


class QM9LOL(torch_geometric.datasets.QM9):
    def __init__(self, root: str, maske: torch.Tensor,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, old_slices = torch.load(self.processed_paths[0])
        # maske = torch.load(os.path.join(self.processed_dir, "masker.pt"))[task]
        # self.slices =
        for lol in old_slices.values():
            print(lol.shape)
        for slice_type in old_slices.values():
            slice_type = slice_type[maske]


    # def process(self):
    #     super().process()
    #     n = len(self) + 1
    #     fordeling = torch.tensor([0.9, 0.005, 0.045, 0.05])
    #     a = torch.multinomial(fordeling, n)
    #     tasks = ['pretrain', 'train', 'val', 'test']
    #     masker = {}
    #     for i in range(4):
    #         task = tasks[i]
    #         maske = a == i
    #         maske[-1] = True
    #         masker[task] = maske
    #     torch.save(masker, "masker.pt")
    #
    # @property
    # def processed_file_names(self) -> List[str]:
    #     return ['data_v3.pt', 'masker.pt']


def make_masker(maske_path, fordeling, n):
    if not fordeling:
        fordeling = torch.tensor([0.9025, 0.0025, 0.045, 0.05])
    a = torch.multinomial(fordeling, n, replacement=True)
    tasks = ['pretrain', 'train', 'val', 'test']
    idxs = {}
    for i in range(4):
        task = tasks[i]
        idxs[task] = torch.where(a == i)[0]
    torch.save(idxs, maske_path)
    return idxs


def byg_QM9(root, task, fordeling: torch.Tensor = None):
    dataset = torch_geometric.datasets.QM9(root)
    maske_path = os.path.join(root, "processed", "masker.pt")
    if os.path.exists(maske_path):
        idxs = torch.load(maske_path)
    else:
        idxs = make_masker(maske_path, fordeling, len(dataset))
    dataset = torch.utils.data.Subset(dataset, idxs[task])
    return dataset


class QM9Selvvejledt(torch_geometric.datasets.QM9):
    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # with open(self.raw_paths[1], 'r') as f:
        #     target = f.read().split('\n')[1:-1]
        #     target = [[float(x) for x in line.split(',')[1:20]]
        #               for line in target]
        #     target = torch.tensor(target, dtype=torch.float)
        #     target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
        #     target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            y = pos[edge_index[0, :], :] - pos[edge_index[1, :], :]
            y = torch.linalg.vector_norm(y, dim=1)
            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y,
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
