import urllib.error

import torch
from torch_geometric.datasets import ZINC
from rdkit import Chem
import pickle
import os
import gzip
import shutil
from torch_geometric.data import Data, download_url
import random
from tqdm import tqdm
import deepchem as dc

def unpack_gzip(gz_file, output_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

class ZINC3D(ZINC):

    def split(self, all_mols):
        random.shuffle(all_mols)
        total_molecules = len(all_mols)
        train_size = int(total_molecules * 1)
        val_size = int(total_molecules * 0.0)
        test_size = total_molecules - train_size - val_size
        split_mols = {}
        split_mols['train'] = random.sample(all_mols, train_size)
        split_mols['val'] = random.sample(list(set(all_mols) - set(split_mols['train'])), val_size)
        split_mols['test'] = random.sample(list(set(all_mols) - set(split_mols['train']) - set(split_mols['val'])), test_size)
        return split_mols

    def process(self):
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        with open(os.path.join(self.raw_dir, "all.pickle"), "rb") as f:
            all_mols = pickle.load(f)
        if self.subset:
            all_mols = random.sample(all_mols, int(len(all_mols)*0.05))


        split_mols = self.split(all_mols)

        for task, mols in split_mols.items():
            data_list = []
            for i, mol in tqdm(enumerate(mols)):
                features = featurizer.featurize(mol)
                x = torch.tensor(features[0].node_features).to(torch.float32)
                edge_attr = torch.tensor(features[0].edge_features).to(torch.float32)
                edge_index = torch.tensor(features[0].edge_index)

                conf = mol.GetConformer()
                pos = conf.GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
                atomic_number = []
                for atom in mol.GetAtoms():
                    atomic_number.append(atom.GetAtomicNum())

                z = torch.tensor(atomic_number, dtype=torch.long)

                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

                data = Data(
                    x=x,
                    z=z,
                    pos=pos,
                    edge_index=edge_index,
                    smiles=smiles,
                    edge_attr=edge_attr,
                    idx=i,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            if len(data_list) > 0:
                data_list = self.collate(data_list)
            torch.save(data_list,
                       os.path.join(self.processed_dir, f'{task}.pt'))

    def download(self):
        with open("data/ZINC-downloader-3D-sdf.gz.uri", 'r') as file:
            urls = file.readlines()
        urls = random.sample(urls, k=int(0.0001*len(urls)))
        filereaders = []
        for url in tqdm(urls):
            url = url.strip()
            try:
                path = download_url(url, self.raw_dir, log=False)
            except urllib.error.HTTPError:
                continue
            unzipped_path = os.path.join(self.raw_dir, os.path.basename(path)[:-3])
            unpack_gzip(path, unzipped_path)

            with Chem.MultithreadedSDMolSupplier(unzipped_path) as sdSupl:
                for mol in sdSupl:
                    if mol is not None:
                        filereaders.append(mol)

            os.remove(path)
            os.remove(unzipped_path)

        print(f"done downloading. You have {len(filereaders)} molecules in total")
        with open(os.path.join(self.raw_dir, "all.pickle"), 'wb') as f:
            pickle.dump(filereaders, f)

    @property
    def raw_file_names(self):
        return ['all.pickle']




if __name__ == "__main__":
    # dataset = ZINC("../ZINC", subset=True)
    # for molekyle in dataset:
    #     # m = Chem.MolFromSmiles(molekyle['smiles'])
    #     break
    # # with open("C:/Users/elleh/OneDrive/speciale/ZINC/raw/train.pickle", "rb") as f:
    # #     atom_dict = pickle.load(f)
    # a, b = torch.load("C:/Users/elleh/OneDrive/speciale/ZINC/subset/processed/train.pt")
    # c = 2
    dataset = ZINC3D("data/zinc", subset=False)
