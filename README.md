# EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction
Welcome!
These files are the source code for our work. (EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction)

## Python file breakdown

### `protein_feature.py`
Builds the graph representation used as model input from a protein structure file.
- Loads a protein in `pdb`, `mol2`, or `sdf` format with RDKit.
- Extracts per-atom features (atomic number, charge, chirality, aromatic/ring flags, degree) and 3D coordinates.
- Extracts bond features (bond type, ring flag, bond length) and creates bidirectional graph edges.
- Calls MSMS to compute molecular surface vertices.
- Maps surface vertices back to nearest atoms and computes geometric descriptors for each vertex:
  - local neighborhood geometry from KNN surface points,
  - global shape features relative to surface centers and atom positions.
- Packs everything into a `torch_geometric.data.Data` object (`x`, `pos`, `edge_index`, `edge_attr`, surface-related tensors) for training/inference.

### `data/0_clean_data.py`
Dataset preprocessing script that splits complexes into protein/ligand files and standardizes dataset layouts.
- Uses PyMOL to split a complex into polymer (protein) and ligand atoms.
- Supports ligand extraction by residue name lists (used by COACH420/HOLO4K mapping files).
- Includes helpers to remove non-standard residues from PDB files.
- Provides dataset-specific cleaning pipelines:
  - `clean_scPDB`
  - `clean_coach420`
  - `clean_holo4k`
  - `clean_PDBbind`
- Writes cleaned files under `1_clean_data/...` directories.

### `models/EquiPocket.py`
Main model definition for EquiPocket.
- Combines three modeling parts:
  1. **Local geometric modeling** from surface descriptors.
  2. **Global structure modeling** via baseline GNN backbones (`Baseline_Models`, configured as GAT+EGNN by default).
  3. **Surface message passing** with `SurfaceEGNN` over surface atoms.
- Optionally applies dense attention weights conditioned on cutoff-distance ratios.
- Outputs:
  - `y_hat`: per-surface-atom binding-site logit/score,
  - `angle`: predicted relative direction vector based on updated node position.

### `models/surface_egnn.py`
Surface-aware E(3)-equivariant message passing layers.
- Defines `MC_E_GCL`, a geometric convolution layer that updates:
  - node features,
  - paired coordinates (atom position + local surface-center anchor).
- Edge geometric features include distances and angular relations between atom and surface-anchor vectors.
- `SurfaceEGNN` stacks multiple `MC_E_GCL` layers and supports dense concatenation of intermediate states.
- Includes utility methods for segment reductions and edge construction.

### `models/baseline_models.py`
Backbone model zoo and chemical feature embeddings used by EquiPocket.
- `embed_atom_chem`: learned embeddings for atom attributes.
- `embed_bond_chem`: learned embeddings for bond attributes.
- `Baseline_Models`: configurable stack that can include:
  - GAT,
  - GCN,
  - GIN,
  - GCN2,
  - SchNet,
  - DimeNet,
  - EGNN.
- In EquiPocket defaults, this module is used to generate global structural node embeddings.

### `models/egnn_clean.py`
Reference EGNN implementation (adapted from the original EGNN repository noted below).
- Implements:
  - `E_GCL` (equivariant graph convolution layer),
  - `EGNN` (stacked E_GCL network),
  - utilities for segment ops and edge construction.
- Performs feature and coordinate updates while preserving E(n)-equivariance.
- Used as a reusable component by `Baseline_Models` when `egnn_depth > 0`.

## Non-code repository files
- `requirements.txt`: Python package dependencies.
- `processed_data/5ei3_protein.pdb`: example cleaned protein.
- `processed_data/5ei3_protein.pkl`: example processed graph output.

## Dependency
 - python: 3.7
 - cuda: 11.6
 - python packages: requirements.txt
 - MSMS: https://ccsb.scripps.edu/msms/


## Datasets
 - scPDB: http://bioinfo-pharma.u-strasbg.fr/scPDB
 - PDBbind: http://www.pdbbind.org.cn
 - COACH420: https://github.com/rdk/p2rank-datasets/tree/master/coach420
 - HOLO4K: https://github.com/rdk/p2rank-datasets/tree/master/holo4k
