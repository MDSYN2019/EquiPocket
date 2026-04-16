# lets generate the dataset for the pdb file then
# generate the accurate scoring of sites that can be scored as a protein-binding sites
from tqdm import tqdm
import torch
import torch.nn.functional as F
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


# Need a traning loop here to train the model


# defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

def train(epochs, model, data) -> None:
    """
    Creating the training loop for the model. We will train the model for a certain number of epochs, and in each epoch, we will iterate through the batches of data, compute the loss, and update the model parameters using backpropagation. After each epoch, we will evaluate the model on the training, validation and test sets and print the accuracy for each set. The training loop will also include a progress bar to track the training progress.
    """    
    for epoch in tqdm(range(epochs)):         
        for step, batch in enumerate(loader, start=1):            
            model.train()    # set the model to training mode, which enables dropout and other training-specific behaviours
            optimizer.zero_grad() # clear the gradients of all optimized parameters
            logits, h = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            logits, _ = model(data.x, data.edge_index)
            pred_class = logits.argmax(dim=1)
                
            print(f"prediction {pred_class}") 
            train_acc = (pred_class[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred_class[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred_class[data.test_mask] == data.y[data.test_mask]).float().mean()
            
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train accuracy: {train_acc:.3f} | "
                f"Val accuracy: {val_acc:.3f} | "
                f"Test accuracy: {test_acc:.3f}"
            )


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
