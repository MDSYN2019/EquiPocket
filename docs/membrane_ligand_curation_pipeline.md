# Curated membrane-protein ligand-binding data pipeline

This document defines a practical, high-confidence curation pipeline for building a training set centered on:

- **Membrane proteins** defined by **RCSB membrane annotations + OPM + mpstruc**
- **Experimentally observed ligand-binding labels** from **BioLiP2**
- Optional enrichment with **PDBbind** affinities and **MemProtMD/OPM** membrane environment features

---

## 1) Scope and target output

Create a structure-level dataset where each row is one curated protein–ligand binding event with:

- canonical structure identifiers (`pdb_id`, `assembly_id`, chains)
- membrane evidence (`rcsb_membrane`, `opm_id`, `mpstruc_class`)
- ligand identity (`het_code`, InChIKey/SMILES where available)
- binding-site residues (BioLiP2 primary labels)
- quality controls (resolution, experimental method, occupancy/altloc flags)
- artifact flags (detergent/lipid/crystallization additive/salt)
- optional affinity (`Kd/Ki/IC50`) and membrane-geometry features

Recommended split unit for ML: **clustered by UniProt/accession or fold family**, not random PDB IDs, to prevent leakage.

---

## 2) Data sources and role assignment

### Primary sources (authoritative for this pipeline)

1. **RCSB membrane annotations + OPM/mpstruc cross-reference**
   - define membrane-protein universe
2. **BioLiP2**
   - define experimentally grounded ligand-binding residues and ligand instances

### Optional enrichments

3. **PDBbind**
   - add affinity labels and cleaned complex files
4. **MemProtMD and OPM geometry**
   - add membrane normal, bilayer depth, insertion/tilt descriptors

### Sources to avoid as primary

- Standalone legacy **PDBTM dump** and **Binding MOAD** should be treated as auxiliary sanity checks only.

---

## 3) Entity model (tables/files to build)

Build a normalized intermediate store before model-ready tensors:

1. `structures`
   - keys: `pdb_id`, `assembly_id`
   - fields: exp method, resolution, deposition/revision dates, organism, polymer entities
2. `membrane_evidence`
   - `pdb_id`, chain/entity ids, source (`rcsb`, `opm`, `mpstruc`), class labels
3. `ligand_events`
   - one row per ligand-binding event from BioLiP2
   - ligand id, chain, residue contacts, binding-site residues
4. `artifact_catalog`
   - dictionary of additive/detergent/common crystallization compounds
5. `quality_flags`
   - occupancy, altloc, missing residues near site, resolution threshold passes
6. `affinity_labels` (optional)
   - merged from PDBbind with units normalized to nM + log-scale value
7. `membrane_features` (optional)
   - OPM/MemProtMD depth/tilt/normal/interfacial descriptors

---

## 4) Curation workflow

## Step A — Build candidate membrane structure universe

1. Pull membrane-annotated PDB entries from RCSB membrane portal/annotation endpoints.
2. Join with OPM and mpstruc mapping tables.
3. Keep entries with evidence from:
   - at least one source (broad mode), or
   - at least two sources (strict mode; recommended for benchmark set).
4. Restrict to protein-containing entries; drop pure nucleic-acid systems unless explicitly needed.

Output: `membrane_candidates`.

## Step B — Intersect with BioLiP2 ligand-binding events

1. Import BioLiP2 events.
2. Map events to PDB IDs and chain/entity IDs.
3. Keep only events whose parent structure is in `membrane_candidates`.
4. Expand residue-level binding labels to a canonical numbering space (PDB auth seq + insertion code, optionally UniProt mapping).

Output: `membrane_biolip_events` (core training labels).

## Step C — Exclude obvious non-training ligands/artifacts

Apply rule-based filtering with retained provenance (`drop_reason`):

1. **Always drop ions and simple buffers/salts** unless a specific project wants metalloproteins.
2. **Drop crystallization additives and common solvents** (e.g., PEG fragments, glycerol-like artifacts).
3. **Handle lipids/detergents separately**:
   - if objective is **small-molecule druggability**, mark as non-target and exclude from positive labels.
   - if objective includes **native lipid sites**, keep but label site type = `lipid`.
4. Remove covalent artifacts unless covalent binding prediction is a goal.
5. Collapse duplicated ligand instances in same biological site to avoid overweighting.

Output: `events_filtered` with `site_type` (`orthosteric`, `allosteric`, `lipid`, `artifact_candidate`, `unknown`).

## Step D — Distinguish orthosteric vs allosteric sites

Use a hierarchical label strategy:

1. If target class is GPCR, use GPCRdb site ontology/mappings.
2. Else infer by annotation heuristics:
   - site overlaps endogenous substrate/cofactor pocket -> `orthosteric_like`
   - distal modulatory pocket -> `allosteric_like`
3. Keep uncertain cases as `unknown` rather than forcing labels.

Output: `site_semantics` table.

## Step E — Structural quality gates

Apply strict thresholds for high-quality subset (and keep relaxed subset for ablations):

- experimental method: prioritize X-ray/cryo-EM with interpretable local density
- structure quality: resolution threshold (project-defined), occupancy checks
- local completeness: no severe missing residues within X Å of site
- alternate conformations: handle altloc by deterministic rule (highest occupancy)

Output: `events_hq` + per-event QC report.

## Step F — Optional affinity merge (PDBbind)

1. Match by PDB ID + ligand identity; verify chain/site consistency.
2. Normalize affinity units and relation operators.
3. Keep source and assay metadata for downstream confidence weighting.

Output: `events_hq_affinity`.

## Step G — Optional membrane-environment feature enrichment

Add features for models that need membrane context:

- depth of binding-site residues relative to bilayer center
- orientation of site vector relative to membrane normal
- transmembrane segment overlap and leaflet assignment
- local lipid exposure descriptors (if available)

Output: `events_hq_context`.

---

## 5) Recommended train/val/test strategy

1. Primary split by **protein family/UniProt cluster** (or stricter sequence identity clusters).
2. Secondary constraint: no same ligand scaffold across train/test for scaffold generalization benchmark.
3. Keep two benchmarks:
   - **Core-HQ** (strict evidence + strict QC)
   - **Extended** (broader evidence + relaxed QC)

---

## 6) Minimal schema for model-ready export

For each sample:

- `sample_id`
- `pdb_id`, `assembly_id`, `chain_id`
- `ligand_id` (+ standardized chemistry id)
- `binding_residues` (list of residue indices)
- `site_type` (`orthosteric`/`allosteric`/`lipid`/`unknown`)
- `is_membrane_protein` + membrane evidence count
- QC fields (`resolution`, `occupancy_ok`, `missing_site_atoms`)
- optional `affinity_nM`, `pAffinity`
- optional membrane geometry vector(s)

Store both:

- residue-level labels
- atom-level contact labels (distance-threshold based) for geometric models

---

## 7) Practical implementation checklist

1. Freeze source versions and retrieval dates.
2. Record every filter as executable rule + count before/after.
3. Persist `drop_reason` and `confidence_score` per sample.
4. Validate a random stratified subset manually (especially detergent/lipid confusion cases).
5. Publish a data card describing known biases:
   - over-representation of popular target classes
   - dependence on crystallographic ligands
   - ambiguity in orthosteric/allosteric labeling outside curated families

---

## 8) Suggested default operating mode

If you need one robust default now:

1. Build membrane universe from **RCSB membrane annotations ∩ (OPM ∪ mpstruc)**.
2. Intersect with **BioLiP2** ligand events.
3. Filter out salts/detergents/crystallization artifacts with conservative whitelist/blacklist rules.
4. Keep strict QC subset for primary training.
5. Add PDBbind affinity and MemProtMD/OPM geometry only for tasks that explicitly need them.

This yields a realistic, high-quality dataset with strong experimental grounding while keeping membrane context explicit.
