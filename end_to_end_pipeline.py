# lets generate the dataset for the pdb file then
# generate the accurate scoring of sites that can be scored as a protein-binding sites
import torch
from protein_feature import get_protein_feature
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.EquiPocket import EquiPocket

msms_path = "/home/sang/Desktop/msms_i86_64Linux2_2.6.1"
protein_file_name = "protein.pdb"
# create the dataset for the protein file
tmp_graph = get_protein_feature(protein_file_name, msms_path=msms_path)
print(tmp_graph)
# create the list of the dataset to be fed for the model
dataset = [tmp_graph, tmp_graph, tmp_graph]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
loader = DataLoader(dataset, batch_size=1, shuffle=False)
model = EquiPocket(
        local_geometric_modeling=False,
        global_structure_modeling="gat_egnn",
        surface_egnn_depth=2,
        dense_attention=False,
        cutoff=6,
        out_depth=2,
        out_features=64,
    ).to(device)


model.eval()
print(f"Running on device: {device}")



with torch.no_grad():
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        y_hat, angle = model(batch)
        print(
            f"batch={step} | y_hat shape={tuple(y_hat.shape)} y_hat = {y_hat} "
            f"| angle shape={tuple(angle.shape) if angle is not None else None}"
        )
