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


## End-to-end pipeline (step by step)
If you are wondering what happens **after** `protein_feature` turns a cleaned protein into a PyG object, this is the full flow used in this repository.

1. **Prepare cleaned protein/ligand files**
   Use `data/0_clean_data.py` to split raw complexes into standardized files (`protein.pdb`, `ligand.pdb`) under `1_clean_data/<dataset>/<sample>/`.
   - `split_protein_ligand(...)` extracts polymer as receptor and ligand atoms (optionally by residue names for COACH420/HOLO4K).
   - `clean_scPDB`, `clean_coach420`, `clean_holo4k`, and `clean_PDBbind` are dataset-specific wrappers.

2. **Build one protein graph with `get_protein_feature(...)`**
   `protein_feature.py` converts one cleaned protein structure into a `torch_geometric.data.Data` object:
   - Parse structure by RDKit (`pdb` / `mol2` / `sdf`).
   - Build atom-level graph tensors (`x`, `pos`, `edge_index`, `edge_attr`).
   - Run MSMS to generate surface vertices (`vert_surface`).
   - Map each surface vertex to nearest atom (`vert_atom`) and mark whether atom is on surface (`atom_in_surface`).
   - Compute 7D surface descriptor (`surface_descriptor`) from local KNN geometry + global shape statistics.
   - Package all tensors into a single PyG `Data` sample.

3. **Batch samples with PyG DataLoader**
   For training/inference on many proteins, store each `Data` object (e.g., as `.pkl`) and load them with a PyTorch Geometric `DataLoader`.
   In batching, key tensors like `x`, `pos`, `edge_index`, and surface tensors are concatenated with batch indices handled by PyG.

4. **Run EquiPocket forward pass**
   `models/EquiPocket.py` consumes the batched `Data` and processes it in three branches:
   - **Global branch**: `Baseline_Models` (`models/baseline_models.py`) embeds atom/bond chemistry and applies selected GNN backbone(s) (default includes GAT + EGNN settings).
   - **Local geometric branch**: MLP stack on `surface_descriptor` to encode local surface geometry.
   - **Surface equivariant branch**: `SurfaceEGNN` (`models/surface_egnn.py`) updates atom/surface-anchor representations with E(3)-equivariant message passing.

5. **Fuse features and predict binding signal**
   EquiPocket concatenates/fuses branch outputs and predicts:
   - `y_hat`: per-surface-associated atom score/logit for binding site likelihood.
   - `angle`: relative direction vector from updated coordinates.

6. **(Training usage) build labels and optimize**
   This repository includes model/data-feature code; your training loop should additionally:
   - build labels from known ligand-contact definitions,
   - compute loss from `y_hat` (and optional geometric terms),
   - backpropagate and update model parameters.

### Minimal example (single protein -> PyG sample)
```python
from protein_feature import get_protein_feature

protein_file = "processed_data/5ei3_protein.pdb"
msms_path = "/path/to/msms/bin"  # folder containing pdb_to_xyzr and msms
sample = get_protein_feature(protein_file, msms_path=msms_path)
print(sample)
```

### DataLoader usage with EquiPocket (multi-protein batching)
The model is designed to receive a **batched PyG `Data` object** from `torch_geometric.loader.DataLoader`.  
A runnable script is provided at `exmaples/dataloader_equipocket_example.py`.
Run it with:

```bash
python exmaples/dataloader_equipocket_example.py
```

The script builds synthetic `Data` samples with the fields EquiPocket expects, batches them with `DataLoader`, and runs a forward pass.
Below is the key model setup used there:

```python
import torch
from models.EquiPocket import EquiPocket

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
```

Practical notes:
- Each dataset item must be a PyG `Data` object containing the fields expected by `EquiPocket` (e.g., atom graph tensors + surface tensors).
- `DataLoader` automatically offsets `edge_index` and other index-based tensors across samples in the batch.
- In training, compute your loss from `y_hat` (and optional geometric terms from `angle`), then backpropagate as usual.

### Tensor meaning quick reference
- `x`: atom feature matrix `[num_atoms, 6]`
- `pos`: centered atom coordinates `[num_atoms, 3]`
- `edge_index`: directed bond edges `[2, num_edges]`
- `edge_attr`: bond features `[num_edges, 3]`
- `atom_in_surface`: whether each atom has assigned surface vertices `[num_atoms]`
- `vert_pos`: unique centered surface points `[num_vertices, 3]`
- `vert_atom`: atom index mapped from each surface vertex `[num_vertices]`
- `surface_descriptor`: 7D geometric descriptor per surface vertex `[num_vertices, 7]`

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
