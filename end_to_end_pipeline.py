"""End-to-end training example for EquiPocket.

This script:
1. Builds a protein graph from ``protein.pdb``.
2. Creates binary labels from ``ligand.pdb`` by marking protein atoms close to the ligand.
3. Aligns labels to EquiPocket output nodes (surface atoms only).
4. Trains EquiPocket with a simple train/val split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from protein_feature import get_protein_feature
from build_ligand_contact_labels import build_ligand_contact_labels
from models.EquiPocket import EquiPocket, get_cutoff_ratio


@dataclass
class TrainConfig:
    msms_path: str = "/home/sang/Desktop/msms_i86_64Linux2_2.6.1"
    protein_file: str = "protein_data_files/protein.pdb"
    ligand_file: str = "protein_data_files/ligand.pdb"
    contact_cutoff: float = 4.0
    epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    val_ratio: float = 0.2
    random_seed: int = 42

def _split_mask(num_nodes: int, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build train/val boolean masks for node-level supervision."""
    if num_nodes < 2:
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        return train_mask, val_mask

    ids = list(range(num_nodes)) # Create a list of node indices from 0 to num_nodes - 1
    rng = random.Random(seed)
    rng.shuffle(ids)

    val_count = max(1, int(num_nodes * val_ratio)) # ensure at least one node is in the validation set, even for small graphs 
    val_ids = set(ids[:val_count]) # take the first val_count indices as validation node and put them in a set for O(1) lookup
    
    mask_idx = torch.arange(num_nodes)
    val_mask = torch.tensor([idx in val_ids for idx in mask_idx.tolist()], dtype=torch.bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def _prepare_graph(cfg: TrainConfig) -> Data:
    """Build graph and attach labels/cutoff features needed by EquiPocket."""
    graph = get_protein_feature(cfg.protein_file, msms_path=cfg.msms_path)  # Build the initial graph with all atoms and features

    # Build labels for all atoms, then keep only surface atoms (same indexing as y_hat).
    _, surface_labels = build_ligand_contact_labels(
        protein_pdb=cfg.protein_file,
        ligand_pdb=cfg.ligand_file,
        contact_cutoff=cfg.contact_cutoff,
        atom_in_surface=graph.atom_in_surface,
    ) # Build binary labels for all atoms, then filter to surface atoms 

    print(f"The surface labels we have is {surface_labels}")

    graph.y = surface_labels.float()

    print(f"the graph structure is {graph}")

    # Dense attention in EquiPocket expects this field.
    graph.cutoff_ratio = get_cutoff_ratio(
        pos=graph.pos,
        cutoff=6,
        surface_egnn_depth=2,
    ).float()

    num_surface_nodes = int((graph.atom_in_surface == 1).sum().item())

    if graph.y.shape[0] != num_surface_nodes:
        raise ValueError(
            "Surface label length does not match the number of surface atoms: "
            f"labels={graph.y.shape[0]} vs surface_atoms={num_surface_nodes}."
        )

    train_mask, val_mask = _split_mask(
        num_nodes=graph.y.shape[0],
        val_ratio=cfg.val_ratio, # use the val_ratio from the config to determine how any nodes to put in the validation set 
        seed=cfg.random_seed, 
    )
    graph.train_mask = train_mask
    graph.val_mask = val_mask

    return graph


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = _prepare_graph(cfg)
    dataset = [graph]
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = EquiPocket(
        local_geometric_modeling=False,
        global_structure_modeling="gat_egnn",
        surface_egnn_depth=2,
        dense_attention=False,
        cutoff=6,
        out_depth=2,
        out_features=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    print(f"Running on device: {device}")
    print(
        f"Surface nodes: {graph.y.numel()}, positives: {int(graph.y.sum().item())}, "
        f"negatives: {int((1 - graph.y).sum().item())}"
    )

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch)
            logits = logits.squeeze(-1)
            labels = batch.y.float()

            loss = F.binary_cross_entropy_with_logits(
                logits[batch.train_mask],
                labels[batch.train_mask],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            eval_batch = graph.to(device)
            logits, _ = model(eval_batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= 0.5).float()

            train_acc = (preds[eval_batch.train_mask] == eval_batch.y[eval_batch.train_mask]).float().mean()
            val_acc = (preds[eval_batch.val_mask] == eval_batch.y[eval_batch.val_mask]).float().mean()

        print(
            f"Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f} | "
            f"Train Acc: {train_acc.item():.3f} | Val Acc: {val_acc.item():.3f}"
        )


if __name__ == "__main__":
    #train(TrainConfig())
    cfg = TrainConfig()
    torch.manual_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = _prepare_graph(cfg)
    dataset = [graph]
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = EquiPocket(
        local_geometric_modeling=False,
        global_structure_modeling="gat_egnn",
        surface_egnn_depth=2,
        dense_attention=False,
        cutoff=6,
        out_depth=2,
        out_features=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    print(f"Running on device: {device}")
    print(
        f"Surface nodes: {graph.y.numel()}, positives: {int(graph.y.sum().item())}, "
        f"negatives: {int((1 - graph.y).sum().item())}"
    )

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(batch)
            logits = logits.squeeze(-1)
            labels = batch.y.float()

            loss = F.binary_cross_entropy_with_logits(
                logits[batch.train_mask],
                labels[batch.train_mask],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            eval_batch = graph.to(device)
            logits, _ = model(eval_batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= 0.5).float()

            train_acc = (preds[eval_batch.train_mask] == eval_batch.y[eval_batch.train_mask]).float().mean()
            val_acc = (preds[eval_batch.val_mask] == eval_batch.y[eval_batch.val_mask]).float().mean()

        print(
            f"Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f} | "
            f"Train Acc: {train_acc.item():.3f} | Val Acc: {val_acc.item():.3f}"
        )
