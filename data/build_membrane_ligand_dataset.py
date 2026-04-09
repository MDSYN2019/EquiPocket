#!/usr/bin/env python3
"""Build a curated membrane-protein ligand-binding dataset.

This script is intentionally modular so you can plug in real source exports from:
- RCSB membrane annotations
- OPM / mpstruc
- BioLiP2
- (optional) PDBbind affinity annotations

Expected inputs are CSV/TSV files from your local preprocessing/export steps.
No network calls are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# Conservative defaults; override with --artifact-json when available.
DEFAULT_ION_CODES = {
    "NA", "K", "CA", "MG", "MN", "ZN", "FE", "CU", "CL", "BR", "IOD",
}
DEFAULT_BUFFER_CODES = {
    "SO4", "PO4", "EDO", "GOL", "MES", "TRS", "HEP", "ACT", "FMT", "DMS",
}
DEFAULT_DETERGENT_CODES = {
    "LDA", "BOG", "DM", "DDM", "OG", "LMT", "C8E", "FOS", "CYM", "SDS",
}
DEFAULT_LIPID_CODES = {
    "POPC", "POPE", "DPPC", "DOPC", "CHL", "CHS", "LPP", "PLM", "OLA", "PEE",
}


@dataclass
class MembraneEvidence:
    pdb_id: str
    chain_id: str
    source: str  # rcsb/opm/mpstruc
    membrane_class: str


@dataclass
class StructureMeta:
    pdb_id: str
    assembly_id: str
    exp_method: str
    resolution: Optional[float]
    uniprot_id: str


@dataclass
class LigandEvent:
    pdb_id: str
    chain_id: str
    ligand_code: str
    ligand_instance_id: str
    binding_residues: str  # e.g. "A:123,A:124"
    raw_site_label: str


@dataclass
class AffinityRecord:
    pdb_id: str
    ligand_code: str
    affinity_type: str
    affinity_value: float
    affinity_unit: str


@dataclass
class CuratedEvent:
    sample_id: str
    pdb_id: str
    assembly_id: str
    chain_id: str
    uniprot_id: str
    ligand_code: str
    ligand_instance_id: str
    binding_residues: str
    site_type: str
    evidence_count: int
    evidence_sources: str
    exp_method: str
    resolution: Optional[float]
    quality_ok: bool
    drop_reason: str
    affinity_nM: Optional[float]
    split: str


def normalize_id(value: str) -> str:
    return (value or "").strip().upper()


def read_delimited(path: Path, delimiter: str) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict((k.strip(), (v or "").strip()) for k, v in row.items()) for row in reader]


def read_table(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".tsv":
        return read_delimited(path, "\t")
    return read_delimited(path, ",")


def parse_float(text: str) -> Optional[float]:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_membrane_evidence(rows: Sequence[Dict[str, str]], source: str) -> List[MembraneEvidence]:
    out: List[MembraneEvidence] = []
    for r in rows:
        out.append(
            MembraneEvidence(
                pdb_id=normalize_id(r.get("pdb_id", "")),
                chain_id=normalize_id(r.get("chain_id", "")),
                source=source,
                membrane_class=r.get("membrane_class", "unknown"),
            )
        )
    return out


def load_structures(rows: Sequence[Dict[str, str]]) -> Dict[str, StructureMeta]:
    out: Dict[str, StructureMeta] = {}
    for r in rows:
        pdb_id = normalize_id(r.get("pdb_id", ""))
        if not pdb_id:
            continue
        out[pdb_id] = StructureMeta(
            pdb_id=pdb_id,
            assembly_id=r.get("assembly_id", "1") or "1",
            exp_method=r.get("exp_method", ""),
            resolution=parse_float(r.get("resolution", "")),
            uniprot_id=normalize_id(r.get("uniprot_id", "")),
        )
    return out


def load_biolip(rows: Sequence[Dict[str, str]]) -> List[LigandEvent]:
    out: List[LigandEvent] = []
    for r in rows:
        pdb_id = normalize_id(r.get("pdb_id", ""))
        if not pdb_id:
            continue
        out.append(
            LigandEvent(
                pdb_id=pdb_id,
                chain_id=normalize_id(r.get("chain_id", "")),
                ligand_code=normalize_id(r.get("ligand_code", "")),
                ligand_instance_id=r.get("ligand_instance_id", ""),
                binding_residues=r.get("binding_residues", ""),
                raw_site_label=r.get("site_label", "unknown"),
            )
        )
    return out


def load_pdbbind(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], AffinityRecord]:
    out: Dict[Tuple[str, str], AffinityRecord] = {}
    for r in rows:
        pdb_id = normalize_id(r.get("pdb_id", ""))
        ligand_code = normalize_id(r.get("ligand_code", ""))
        v = parse_float(r.get("affinity_value", ""))
        if not pdb_id or not ligand_code or v is None:
            continue
        rec = AffinityRecord(
            pdb_id=pdb_id,
            ligand_code=ligand_code,
            affinity_type=(r.get("affinity_type", "") or "").upper(),
            affinity_value=v,
            affinity_unit=(r.get("affinity_unit", "") or "").lower(),
        )
        out[(pdb_id, ligand_code)] = rec
    return out


def to_nanomolar(value: float, unit: str) -> Optional[float]:
    u = unit.lower()
    if u in {"nm", "nanomolar", "nм"}:
        return value
    if u in {"um", "μm", "micromolar"}:
        return value * 1e3
    if u in {"mm", "millimolar"}:
        return value * 1e6
    if u in {"pm", "picomolar"}:
        return value * 1e-3
    return None


def load_artifact_codes(path: Optional[Path]) -> Dict[str, Set[str]]:
    if not path:
        return {
            "ions": set(DEFAULT_ION_CODES),
            "buffers": set(DEFAULT_BUFFER_CODES),
            "detergents": set(DEFAULT_DETERGENT_CODES),
            "lipids": set(DEFAULT_LIPID_CODES),
        }
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "ions": set(map(normalize_id, raw.get("ions", []))),
        "buffers": set(map(normalize_id, raw.get("buffers", []))),
        "detergents": set(map(normalize_id, raw.get("detergents", []))),
        "lipids": set(map(normalize_id, raw.get("lipids", []))),
    }


def aggregate_evidence(evidence_rows: Iterable[MembraneEvidence]) -> Dict[Tuple[str, str], Set[str]]:
    by_chain: Dict[Tuple[str, str], Set[str]] = {}
    for e in evidence_rows:
        key = (e.pdb_id, e.chain_id)
        by_chain.setdefault(key, set()).add(e.source)
    return by_chain


def classify_site(raw_site_label: str, ligand_code: str, artifacts: Dict[str, Set[str]]) -> Tuple[str, str]:
    label = (raw_site_label or "").strip().lower()

    if ligand_code in artifacts["ions"]:
        return "artifact_candidate", "ion"
    if ligand_code in artifacts["buffers"]:
        return "artifact_candidate", "buffer_or_solvent"
    if ligand_code in artifacts["detergents"]:
        return "artifact_candidate", "detergent"
    if ligand_code in artifacts["lipids"]:
        return "lipid", "native_lipid_or_lipid_like"

    if "orthosteric" in label:
        return "orthosteric", "annotated_orthosteric"
    if "allosteric" in label:
        return "allosteric", "annotated_allosteric"
    return "unknown", "no_high_confidence_site_semantics"


def passes_quality(meta: Optional[StructureMeta], max_resolution: float) -> bool:
    if meta is None:
        return False
    if meta.exp_method == "":
        return False
    if meta.resolution is None:
        return False
    return meta.resolution <= max_resolution


def make_sample_id(pdb_id: str, chain_id: str, ligand_instance_id: str, ligand_code: str) -> str:
    raw = f"{pdb_id}:{chain_id}:{ligand_instance_id}:{ligand_code}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"sample_{digest}"


def assign_split(uniprot_id: str, seed: str) -> str:
    key = (uniprot_id or "UNKNOWN") + "|" + seed
    score = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if score < 80:
        return "train"
    if score < 90:
        return "val"
    return "test"


def curate_dataset(
    membrane_evidence: Sequence[MembraneEvidence],
    structures: Dict[str, StructureMeta],
    biolip_events: Sequence[LigandEvent],
    artifacts: Dict[str, Set[str]],
    min_evidence_sources: int,
    max_resolution: float,
    split_seed: str,
    affinity: Optional[Dict[Tuple[str, str], AffinityRecord]] = None,
) -> List[CuratedEvent]:
    affinity = affinity or {}
    evidence_map = aggregate_evidence(membrane_evidence)
    out: List[CuratedEvent] = []

    for ev in biolip_events:
        key = (ev.pdb_id, ev.chain_id)
        sources = evidence_map.get(key, set())

        # Fallback: if chain-level map missing, allow any chain for that PDB.
        if not sources:
            candidates = [s for (pdb, _chain), s in evidence_map.items() if pdb == ev.pdb_id]
            for s in candidates:
                sources.update(s)

        if len(sources) < min_evidence_sources:
            continue

        meta = structures.get(ev.pdb_id)
        quality_ok = passes_quality(meta, max_resolution)
        site_type, drop_reason = classify_site(ev.raw_site_label, ev.ligand_code, artifacts)

        # Conservative default: artifact candidates are excluded from positive set.
        if site_type == "artifact_candidate":
            continue

        aff = affinity.get((ev.pdb_id, ev.ligand_code))
        aff_nm = to_nanomolar(aff.affinity_value, aff.affinity_unit) if aff else None

        assembly_id = meta.assembly_id if meta else "1"
        uniprot_id = meta.uniprot_id if meta else ""

        out.append(
            CuratedEvent(
                sample_id=make_sample_id(ev.pdb_id, ev.chain_id, ev.ligand_instance_id, ev.ligand_code),
                pdb_id=ev.pdb_id,
                assembly_id=assembly_id,
                chain_id=ev.chain_id,
                uniprot_id=uniprot_id,
                ligand_code=ev.ligand_code,
                ligand_instance_id=ev.ligand_instance_id,
                binding_residues=ev.binding_residues,
                site_type=site_type,
                evidence_count=len(sources),
                evidence_sources=",".join(sorted(sources)),
                exp_method=meta.exp_method if meta else "",
                resolution=meta.resolution if meta else None,
                quality_ok=quality_ok,
                drop_reason=drop_reason,
                affinity_nM=aff_nm,
                split=assign_split(uniprot_id, split_seed),
            )
        )

    return out


def write_csv(path: Path, rows: Sequence[CuratedEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write header-only file for pipeline stability.
        fieldnames = [f.name for f in CuratedEvent.__dataclass_fields__.values()]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(path: Path, rows: Sequence[CuratedEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate membrane-protein ligand-binding events.")
    p.add_argument("--rcsb", type=Path, required=True, help="CSV/TSV with columns: pdb_id, chain_id, membrane_class")
    p.add_argument("--opm", type=Path, required=True, help="CSV/TSV with columns: pdb_id, chain_id, membrane_class")
    p.add_argument("--mpstruc", type=Path, required=True, help="CSV/TSV with columns: pdb_id, chain_id, membrane_class")
    p.add_argument("--structures", type=Path, required=True, help="CSV/TSV with columns: pdb_id, assembly_id, exp_method, resolution, uniprot_id")
    p.add_argument("--biolip2", type=Path, required=True, help="CSV/TSV with columns: pdb_id, chain_id, ligand_code, ligand_instance_id, binding_residues, site_label")
    p.add_argument("--pdbbind", type=Path, help="Optional CSV/TSV with columns: pdb_id, ligand_code, affinity_type, affinity_value, affinity_unit")
    p.add_argument("--artifact-json", type=Path, help="Optional JSON file with arrays: ions, buffers, detergents, lipids")
    p.add_argument("--min-evidence-sources", type=int, default=2, help="Minimum distinct membrane evidence sources to keep a sample")
    p.add_argument("--max-resolution", type=float, default=3.5, help="Maximum crystal/cryo-EM resolution (Å)")
    p.add_argument("--split-seed", type=str, default="equipocket_membrane_v1")
    p.add_argument("--out-csv", type=Path, default=Path("processed_data/membrane_ligand_curated.csv"))
    p.add_argument("--out-jsonl", type=Path, default=Path("processed_data/membrane_ligand_curated.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rcsb = load_membrane_evidence(read_table(args.rcsb), source="rcsb")
    opm = load_membrane_evidence(read_table(args.opm), source="opm")
    mp = load_membrane_evidence(read_table(args.mpstruc), source="mpstruc")
    membrane = [*rcsb, *opm, *mp]

    structures = load_structures(read_table(args.structures))
    biolip = load_biolip(read_table(args.biolip2))
    artifacts = load_artifact_codes(args.artifact_json)

    affinity = None
    if args.pdbbind:
        affinity = load_pdbbind(read_table(args.pdbbind))

    curated = curate_dataset(
        membrane_evidence=membrane,
        structures=structures,
        biolip_events=biolip,
        artifacts=artifacts,
        min_evidence_sources=args.min_evidence_sources,
        max_resolution=args.max_resolution,
        split_seed=args.split_seed,
        affinity=affinity,
    )

    write_csv(args.out_csv, curated)
    write_jsonl(args.out_jsonl, curated)

    summary = {
        "n_membrane_evidence_rows": len(membrane),
        "n_structures": len(structures),
        "n_biolip_events": len(biolip),
        "n_curated_events": len(curated),
        "out_csv": str(args.out_csv),
        "out_jsonl": str(args.out_jsonl),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
