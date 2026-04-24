#!/usr/bin/env python
"""Populate therapeutic-tool caches for the official_v15 dataset.

This script appends missing raw-SMILES entries for:
  - metadata cache: ``cache/tdc_metadata_consolidated.csv``
  - functional-group cache: ``cache/fg_cache.jsonl``
  - safety cache: ``cache/safety_cache.jsonl``

It intentionally keys entries by the exact input SMILES from the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = ROOT / "openrlhf" / "tools" / "therapeutic_tools"
CACHE_DIR = TOOLS_DIR / "cache"
DATASET_DIR = ROOT / "data" / "tdc" / "official_v15_dataset"
FEATURE_CACHE_DIR = ROOT / "ml_experiments" / "feature_cache_official_v15"

METADATA_CACHE = CACHE_DIR / "tdc_metadata_consolidated.csv"
FG_CACHE = CACHE_DIR / "fg_cache.jsonl"
SAFETY_CACHE = CACHE_DIR / "safety_cache.jsonl"
RING_SYSTEMS_CACHE = CACHE_DIR / "ring_systems_cache.jsonl"
V17_MOLECULAR_PROFILE_CACHE = CACHE_DIR / "v17_molecular_profile_cache.jsonl"
V17_IONIZATION_AND_SOLUBILITY_CACHE = CACHE_DIR / "v17_ionization_and_solubility_cache.jsonl"
V17_STRUCTURE_AND_TOPOLOGY_CACHE = CACHE_DIR / "v17_structure_and_topology_cache.jsonl"

REQUIRED_METADATA_COLUMNS = [
    "MolWt",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "BertzCT",
    "HallKierAlpha",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeterocycles",
    "most_acidic_pka",
    "most_basic_pka",
    "f_neutral_7_4",
    "minimol_solubility_log_mol_L",
]

FEATURE_TO_METADATA_COLUMN = {
    "smiles": "Drug",
    "minimol_log_s": "minimol_solubility_log_mol_L",
}


def collect_dataset_smiles(dataset_dir: Path) -> list[str]:
    smiles: list[str] = []
    seen: set[str] = set()
    for csv_path in sorted(dataset_dir.glob("*/*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("Drug") or row.get("smiles") or row.get("SMILES")
                if not smi or smi in seen:
                    continue
                seen.add(smi)
                smiles.append(smi)
    return smiles


def load_jsonl_keys(path: Path, key: str = "smiles") -> set[str]:
    values: set[str] = set()
    if not path.exists():
        return values
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            value = entry.get(key)
            if isinstance(value, str):
                values.add(value)
    return values


def load_safety_cache_variants(path: Path) -> set[tuple[str, bool]]:
    values: set[tuple[str, bool]] = set()
    if not path.exists():
        return values
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            smiles = entry.get("smiles")
            include_smarts = entry.get("include_smarts", True)
            if isinstance(smiles, str) and isinstance(include_smarts, bool):
                values.add((smiles, include_smarts))
    return values


def build_missing_metadata_rows(missing_smiles: set[str], metadata_columns: list[str]) -> list[dict]:
    if not missing_smiles:
        return []

    collected: dict[str, dict] = {}
    feature_files = sorted(FEATURE_CACHE_DIR.glob("*/*_all_tools_features.csv"))
    for path in feature_files:
        if len(collected) == len(missing_smiles):
            break
        use_cols = ["smiles"] + [c for c in REQUIRED_METADATA_COLUMNS if c != "minimol_solubility_log_mol_L"]
        use_cols.append("minimol_log_s")
        available_cols = None
        try:
            header = pd.read_csv(path, nrows=0)
            available_cols = [c for c in use_cols if c in header.columns]
        except Exception:
            continue
        if not available_cols or "smiles" not in available_cols:
            continue
        for chunk in pd.read_csv(path, usecols=available_cols, chunksize=2048):
            chunk = chunk[chunk["smiles"].isin(missing_smiles)]
            if chunk.empty:
                continue
            for _, row in chunk.iterrows():
                smiles = str(row["smiles"])
                if smiles in collected:
                    continue
                payload = {col: pd.NA for col in metadata_columns}
                payload["Drug"] = smiles
                for feature_col, value in row.items():
                    target_col = FEATURE_TO_METADATA_COLUMN.get(feature_col, feature_col)
                    if target_col in payload:
                        payload[target_col] = value
                collected[smiles] = payload
            if len(collected) == len(missing_smiles):
                break

    unresolved = sorted(missing_smiles - set(collected))
    if unresolved:
        raise RuntimeError(
            f"Could not source metadata rows for {len(unresolved)} official_v15 SMILES from {FEATURE_CACHE_DIR}"
        )
    return [collected[smi] for smi in sorted(collected)]


def append_metadata_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.read_csv(METADATA_CACHE)
    metadata_columns = list(df.columns)
    additions = pd.DataFrame(rows, columns=metadata_columns)
    merged = pd.concat([df, additions], ignore_index=True)
    merged.to_csv(METADATA_CACHE, index=False)
    return len(additions)


def append_fg_cache(smiles_list: Iterable[str]) -> int:
    smiles_list = list(smiles_list)
    if not smiles_list:
        return 0
    from openrlhf.tools.therapeutic_tools.legacy_tools.AccFG import concise_fg_description

    written = 0
    with FG_CACHE.open("a") as out:
        for idx, smiles in enumerate(smiles_list, 1):
            try:
                fg_text = concise_fg_description(smiles)
            except Exception as e:
                fg_text = f"Error: {e}"
            out.write(json.dumps({"smiles": smiles, "fg": fg_text}, ensure_ascii=False) + "\n")
            written += 1
            if idx % 50 == 0:
                print(f"fg cached {idx}/{len(smiles_list)}")
    return written


def append_safety_cache(smiles_variants: Iterable[tuple[str, bool]]) -> int:
    smiles_variants = list(smiles_variants)
    if not smiles_variants:
        return 0
    from openrlhf.tools.therapeutic_tools.safety import screen_safety

    written = 0
    with SAFETY_CACHE.open("a") as out:
        for idx, (smiles, include_smarts) in enumerate(smiles_variants, 1):
            try:
                result = screen_safety(smiles, include_smarts=include_smarts)
            except Exception as e:
                result = f"Structural Alerts: Error screening molecule - {e}"
            out.write(
                json.dumps(
                    {
                        "smiles": smiles,
                        "include_smarts": include_smarts,
                        "result": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if idx % 50 == 0:
                print(f"safety cached {idx}/{len(smiles_variants)}")
    return written


def append_string_cache(smiles_list: Iterable[str], cache_path: Path, compute_fn, label: str) -> int:
    smiles_list = list(smiles_list)
    if not smiles_list:
        return 0

    written = 0
    with cache_path.open("a") as out:
        for idx, smiles in enumerate(smiles_list, 1):
            try:
                result = compute_fn(smiles)
            except Exception as e:
                result = f"Error: {e}"
            out.write(json.dumps({"smiles": smiles, "result": result}, ensure_ascii=False) + "\n")
            written += 1
            if idx % 50 == 0:
                print(f"{label} cached {idx}/{len(smiles_list)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate metadata/FG/safety caches for official_v15 raw SMILES")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-fg", action="store_true")
    parser.add_argument("--skip-safety", action="store_true")
    parser.add_argument("--skip-ring-systems", action="store_true")
    parser.add_argument("--skip-v17-groups", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    smiles = collect_dataset_smiles(dataset_dir)
    print(f"Collected {len(smiles)} unique raw SMILES from {dataset_dir}")

    if not args.skip_metadata:
        meta_df = pd.read_csv(METADATA_CACHE, nrows=0)
        metadata_columns = list(meta_df.columns)
        metadata_done = set(pd.read_csv(METADATA_CACHE, usecols=["Drug"])["Drug"].astype(str))
        metadata_missing = {s for s in smiles if s not in metadata_done}
        rows = build_missing_metadata_rows(metadata_missing, metadata_columns)
        added = append_metadata_rows(rows)
        print(f"metadata appended {added} rows")

    if not args.skip_fg:
        fg_done = load_jsonl_keys(FG_CACHE)
        fg_missing = [s for s in smiles if s not in fg_done]
        added = append_fg_cache(fg_missing)
        print(f"fg appended {added} rows")

    if not args.skip_safety:
        safety_done = load_safety_cache_variants(SAFETY_CACHE)
        safety_missing = [
            (s, include_smarts)
            for s in smiles
            for include_smarts in (True, False)
            if (s, include_smarts) not in safety_done
        ]
        added = append_safety_cache(safety_missing)
        print(f"safety appended {added} rows")

    if not args.skip_ring_systems:
        from openrlhf.tools.therapeutic_tools.ring_systems import analyze_ring_systems

        ring_done = load_jsonl_keys(RING_SYSTEMS_CACHE)
        ring_missing = [s for s in smiles if s not in ring_done]
        added = append_string_cache(ring_missing, RING_SYSTEMS_CACHE, analyze_ring_systems, "ring_systems")
        print(f"ring_systems appended {added} rows")

    if not args.skip_v17_groups:
        from openrlhf.tools.therapeutic_tools.v17 import (
            _compute_ionization_and_solubility_v17_uncached,
            _compute_molecular_profile_v17_uncached,
            _compute_structure_and_topology_v17_uncached,
        )

        group_specs = [
            ("v17_molecular_profile", V17_MOLECULAR_PROFILE_CACHE, _compute_molecular_profile_v17_uncached),
            ("v17_ionization_and_solubility", V17_IONIZATION_AND_SOLUBILITY_CACHE, _compute_ionization_and_solubility_v17_uncached),
            ("v17_structure_and_topology", V17_STRUCTURE_AND_TOPOLOGY_CACHE, _compute_structure_and_topology_v17_uncached),
        ]
        for label, path, compute_fn in group_specs:
            done = load_jsonl_keys(path)
            missing = [s for s in smiles if s not in done]
            added = append_string_cache(missing, path, compute_fn, label)
            print(f"{label} appended {added} rows")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
