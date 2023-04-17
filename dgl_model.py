import numpy as np
import pandas as pd
import urllib.request
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors
import torch
import dgllife
from dgllife.utils import mol_to_complete_graph
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import *
m = Chem.MolFromSmiles('CC1(C)\C(=C/C=C(Br)/C=C/C2=[N+](CCCC3=CC=CC=C3)C3=C(C4=C(C=CC=C4)C=C3)C2(C)C)N(CCCC2=CC=CC=C2)C2=C1C1=C(C=CC=C1)C=C2')
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='h')
mol_to_complete_graph(m, node_featurizer=node_featurizer)


import threading
import numpy as np
import pandas as pd
import torch_geometric as pyg
from random import Random

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from torch_geometric import data
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from typing import Dict, Callable, List, Union, Optional, Iterator