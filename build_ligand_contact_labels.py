"""Utilities to build ligand-contact labels for EquiPocket training.

Given `protein.pdb` and `ligand.pdb`, this script creates binary labels by
marking protein atoms whose minimum distance to any ligand atom is <= cutoff.

The output can be generated for:
1) all protein atoms, or
2) only `atom_in_surface == 1` atoms (the same indexing used by EquiPocket's
   `y_hat` output).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple

import torch
from rdkit import Chem


DEFAULT_CONTACT_CUTOFF = 4.0


def _load_pdb_atom_positions(pdb_path: str) -> torch.Tensor:
    """Load atom coordinates from a PDB file using RDKit.

    Returns a float tensor with shape [num_atoms, 3].
    """
    mol = Chem.MolFromPDBFile(pdb_path)
    if mol is None:
        raise ValueError(f"Could not parse PDB file: {pdb_path}")

    conf = mol.GetConformer()
    coords = []
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append([pos.x, pos.y, pos.z])

    if len(coords) == 0:
        raise ValueError(f"No atoms found in file: {pdb_path}")

    return torch.tensor(coords, dtype=torch.float32)


def build_ligand_contact_labels(
    protein_pdb: str,
    ligand_pdb: str,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
    atom_in_surface: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ligand-contact labels.

    Args:
        protein_pdb: path to protein structure (`protein.pdb`).
        ligand_pdb: path to ligand structure (`ligand.pdb`).
        contact_cutoff: distance cutoff in Å for defining contact.
        atom_in_surface: optional [num_protein_atoms] mask from `graph_data.atom_in_surface`.

    Returns:
        labels_all_atoms: binary labels for every protein atom, shape [N].
        labels_surface_atoms: binary labels for only atoms where atom_in_surface==1.
            If `atom_in_surface` is None, this is identical to labels_all_atoms.
    """
    if contact_cutoff <= 0:
        raise ValueError("contact_cutoff must be positive.")

    protein_pos = _load_pdb_atom_positions(protein_pdb) # [num_protein_atoms, 3] 
    ligand_pos = _load_pdb_atom_positions(ligand_pdb) # [num_ligand_atoms, 3]

    # Pairwise distances: [num_protein_atoms, num_ligand_atoms]
    dists = torch.cdist(protein_pos, ligand_pos) # Compute the pairwise distances between protein and ligand atoms 
    min_dist_per_protein_atom = dists.min(dim=1).values  # get the minimum distance to any ligand atom for each protein atom
    labels_all_atoms = (min_dist_per_protein_atom <= contact_cutoff).float() # binary labels: 1 if in contact, 0 otherwise - returns a float tensor of shape [num_protein_atoms]

    if atom_in_surface is None: # if no surface mask is provided, we consider all atoms as surface atoms for the purpose of labeling
        labels_surface_atoms = labels_all_atoms
    else: # if a surface mask is provided, we filter the labels to only include those atoms that are marked as surface atoms (atoms_in_surface == 1)        
        atom_in_surface = atom_in_surface.reshape(-1) 
        if atom_in_surface.shape[0] != labels_all_atoms.shape[0]: 
            raise ValueError(
                "atom_in_surface length mismatch: "
                f"got {atom_in_surface.shape[0]}, expected {labels_all_atoms.shape[0]}"
            )
        surface_mask = atom_in_surface == 1
        labels_surface_atoms = labels_all_atoms[surface_mask] # filter the labels to only include the surface atoms based on the label_all_atoms and the surtface mask
    
    return labels_all_atoms, labels_surface_atoms # return the labels for all atoms and the labels for only the surface atoms (if a surface mask is provided)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ligand-contact labels from protein/ligand PDB files.")
    parser.add_argument("--protein", required=True, help="Path to protein.pdb")
    parser.add_argument("--ligand", required=True, help="Path to ligand.pdb")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CONTACT_CUTOFF,
        help="Contact distance cutoff in Angstrom (default: 4.0)",
    )
    parser.add_argument(
        "--graph-data",
        default=None,
        help=(
            "Optional pickle file containing a PyG Data object with atom_in_surface. "
            "If provided, also outputs labels aligned with y_hat indexing."
        ),
    )
    parser.add_argument(
        "--out",
        default="ligand_contact_labels.pt",
        help="Output .pt file path.",
    )
    args = parser.parse_args()

    atom_in_surface = None
    if args.graph_data is not None:
        with open(args.graph_data, "rb") as f:
            graph_data = pickle.load(f)
        if not hasattr(graph_data, "atom_in_surface"):
            raise ValueError("graph-data does not include `atom_in_surface`.")
        atom_in_surface = graph_data.atom_in_surface

    labels_all, labels_surface = build_ligand_contact_labels(
        protein_pdb=args.protein,
        ligand_pdb=args.ligand,
        contact_cutoff=args.cutoff,
        atom_in_surface=atom_in_surface,
    )

    payload = {
        "protein_pdb": str(Path(args.protein).resolve()),
        "ligand_pdb": str(Path(args.ligand).resolve()),
        "contact_cutoff": args.cutoff,
        "labels_all_atoms": labels_all,
        "labels_surface_atoms": labels_surface,
        "num_positive_all_atoms": int(labels_all.sum().item()),
        "num_positive_surface_atoms": int(labels_surface.sum().item()),
    }
    torch.save(payload, args.out)

    print(f"Saved labels to: {args.out}")
    print(f"All atoms: {labels_all.shape[0]} (positives: {int(labels_all.sum().item())})")
    print(
        f"Surface atoms: {labels_surface.shape[0]} "
        f"(positives: {int(labels_surface.sum().item())})"
    )


if __name__ == "__main__":
    main()
