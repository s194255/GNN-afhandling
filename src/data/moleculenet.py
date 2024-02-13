import torch
from torch_geometric.datasets import ZINC, QM9
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

if __name__ == "__main__":
    dataset = QM9("../QM9")
    # a = torch.load("C:/Users/elleh/OneDrive/speciale/QM9/raw/qm9_v3.pt", weights_only=True)
    for molekyle in dataset:
        m = Chem.MolFromSmiles(molekyle['smiles'])
        # Draw.MolToImage(m).show()
        print(Chem.MolToMolBlock(m))
        m = Chem.AddHs(m)
        Chem.AllChem.EmbedMolecule(m)
        print(Chem.MolToMolBlock(m))

        for i, atom in enumerate(m.GetAtoms()):
            positions = m.GetConformer().GetAtomPosition(i)
            print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        break