import torch
from torchdrug import data, datasets

dataset = datasets.ClinTox("./TorchDrugModel/")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

from torchdrug import core, models, tasks, utils
model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))


optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=1024)
solver.train(num_epoch=100)
solver.evaluate("valid")




mol = data.Molecule.from_smiles(["C1=CC=CC=C1", "C1=CC=CC=C1"], atom_feature="default")
mols = [mol, mol]
model = models.GIN(input_dim=mol.node_feature.shape[-1], hidden_dims=[128, 128])
output = model(mol, mol.node_feature.float())


mols = data.Molecule.from_smiles("C1=CC=CC=C1", atom_feature="default")

from torchdrug.data import MoleculeDataset
MD = MoleculeDataset()
a = MD.load_csv(csv_file='/mnt/home/linjie/projects/solubility/data/ESL-clean-final-20220704/ESL-clean-final-20220704_smiles_pLogS.csv', smiles_field='smiles', target_fields='pLogS')