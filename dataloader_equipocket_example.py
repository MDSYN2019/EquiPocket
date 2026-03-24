"""Minimal runnable DataLoader -> EquiPocket forward example.

This script builds small synthetic PyG Data samples that match the fields
expected by EquiPocket, batches them with torch_geometric.loader.DataLoader,
and runs a forward pass.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.EquiPocket import EquiPocket


def make_toy_sample(num_nodes: int = 8) -> Data:
    # Atom features expected by embed_atom_chem:
    # [atomic_num, formal_charge, chiral_tag, aromatic_flag, ring_flag, degree_or_scalar]
    x = torch.zeros((num_nodes, 6), dtype=torch.float)
    x[:, 0] = 6  # carbon-like atomic number token
    x[:, 1] = 0  # charge token
    x[:, 2] = 0  # chirality token
    x[:, 3] = torch.randint(0, 2, (num_nodes,), dtype=torch.long).float()
    x[:, 4] = torch.randint(0, 2, (num_nodes,), dtype=torch.long).float()
    x[:, 5] = torch.randint(1, 4, (num_nodes,), dtype=torch.long).float()

    # Simple 3D coordinates with nearby points so radius graphs are not empty.
    pos = torch.randn((num_nodes, 3), dtype=torch.float) * 0.3

    # Bidirectional chain edges.
    src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    dst = src + 1
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])],
        dim=0,
    )

    # Edge attributes expected by embed_bond_chem: [bond_type, in_ring, bond_length_scalar]
    num_edges = edge_index.shape[1]
    edge_attr = torch.zeros((num_edges, 3), dtype=torch.float)
    edge_attr[:, 0] = 1  # single bond token
    edge_attr[:, 1] = 0  # ring flag token
    edge_attr[:, 2] = 1.0  # placeholder scalar

    # Mark a subset of atoms as surface-associated (must include >=2 per graph).
    atom_in_surface = torch.zeros((num_nodes,), dtype=torch.long)
    atom_in_surface[: max(2, num_nodes // 2)] = 1

    # Surface anchor coordinates paired with each atom (same length as num_nodes).
    surface_center_pos = pos + 0.05 * torch.randn_like(pos)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        atom_in_surface=atom_in_surface,
        surface_center_pos=surface_center_pos,
    )


def main() -> None:
    dataset = [make_toy_sample(8), make_toy_sample(10), make_toy_sample(12)]
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # local_geometric_modeling=False here keeps the minimal synthetic sample small
    # (no surface_descriptor/vert_batch tensors required).
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
                f"batch={step} | y_hat shape={tuple(y_hat.shape)} "
                f"| angle shape={tuple(angle.shape) if angle is not None else None}"
            )


if __name__ == "__main__":
    main()
