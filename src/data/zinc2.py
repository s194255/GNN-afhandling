import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer

featurizer = MolGraphConvFeaturizer(use_edges=True)
# dataset_dc = dc.molnet.load_zinc15(featurizer=featurizer, data_dir='zinc/processed', save_dir='raw')
dataset_dc = dc.molnet.load_zinc15(featurizer='OneHot', save_dir='raw')