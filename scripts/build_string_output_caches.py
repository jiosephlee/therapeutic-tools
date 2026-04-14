#!/usr/bin/env python
"""
Build precomputed string-output caches for therapeutic tools over TDC molecules.

Default outputs:
  - openrlhf/tools/therapeutic_tools/cache/safety_cache.jsonl
  - openrlhf/tools/therapeutic_tools/cache/three_d_cache.jsonl

Each line is a JSON object keyed by canonical SMILES.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data" / "tdc" / "deduplicated_canonicalized"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
SAFETY_CACHE = CACHE_DIR / "safety_cache.jsonl"
THREE_D_CACHE = CACHE_DIR / "three_d_cache.jsonl"


def collect_smiles() -> list[str]:
    seen: set[str] = set()
    all_smiles: list[str] = []
    for path in sorted(glob.glob(str(DATA_DIR / "*" / "*.csv"))):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row.get("Drug") or row.get("smiles") or row.get("SMILES")
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    all_smiles.append(smiles)
    return all_smiles


def load_done(path: Path, include_epsa: bool | None = None) -> set[tuple[str, bool] | str]:
    done: set[tuple[str, bool] | str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            smiles = entry.get("smiles")
            if not isinstance(smiles, str):
                continue
            if include_epsa is None:
                done.add(smiles)
            else:
                epsa = entry.get("include_epsa")
                if isinstance(epsa, bool):
                    done.add((smiles, epsa))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-safety", action="store_true")
    parser.add_argument("--skip-three-d", action="store_true")
    parser.add_argument("--only-three-d-no-epsa", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openrlhf")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from rdkit import Chem
    from openrlhf.tools.therapeutic_tools.safety import screen_safety
    from openrlhf.tools.therapeutic_tools.three_d import get_3d_properties

    smiles_list = collect_smiles()
    print(f"Collected {len(smiles_list)} unique SMILES")

    if not args.skip_safety:
        done = load_done(SAFETY_CACHE)
        mode = "a" if SAFETY_CACHE.exists() else "w"
        with SAFETY_CACHE.open(mode, buffering=1) as out:
            remaining = 0
            for idx, smiles in enumerate(smiles_list, 1):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if canonical in done:
                    continue
                remaining += 1
                result = screen_safety(canonical)
                out.write(json.dumps({"smiles": canonical, "result": result}) + "\n")
                out.flush()
                done.add(canonical)
                if remaining % 100 == 0:
                    print(f"safety cached {remaining}")
        print(f"safety cache size {len(done)}")

    if not args.skip_three_d:
        include_epsa_values = [False] if args.only_three_d_no_epsa else [False, True]
        done = load_done(THREE_D_CACHE, include_epsa=False) | load_done(THREE_D_CACHE, include_epsa=True)
        mode = "a" if THREE_D_CACHE.exists() else "w"
        with THREE_D_CACHE.open(mode, buffering=1) as out:
            remaining = 0
            for idx, smiles in enumerate(smiles_list, 1):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                for include_epsa in include_epsa_values:
                    key = (canonical, include_epsa)
                    if key in done:
                        continue
                    remaining += 1
                    result = get_3d_properties(canonical, include_epsa=include_epsa)
                    out.write(
                        json.dumps(
                            {"smiles": canonical, "include_epsa": include_epsa, "result": result}
                        )
                        + "\n"
                    )
                    out.flush()
                    done.add(key)
                    if remaining % 100 == 0:
                        print(f"three_d cached {remaining}")
        print(f"three_d cache size {len(done)}")


if __name__ == "__main__":
    main()
