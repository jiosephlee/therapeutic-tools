"""Update tdc_metadata_consolidated.csv with per-site pKa JSON columns.

Adds two columns:
  - acid_sites_json: JSON string mapping atom_map_number -> pKa for each acidic site
  - base_sites_json: JSON string mapping atom_map_number -> pKa for each basic site

Runs MolGpKa on all molecules in the cache. Takes ~70 minutes for 45k molecules.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import pandas as pd
import numpy as np
from rdkit import Chem

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "tdc_metadata_consolidated.csv")


def main():
    df = pd.read_csv(CACHE_PATH)
    print(f"Loaded {len(df)} molecules from cache")

    # Skip if already done
    if "acid_sites_json" in df.columns and "base_sites_json" in df.columns:
        n_existing = df["acid_sites_json"].notna().sum()
        print(f"acid_sites_json already exists with {n_existing} non-null entries")
        if n_existing == len(df):
            print("Already fully populated, nothing to do.")
            return

    from openrlhf.tools.therapeutic_tools.legacy_tools.pka_related_tools import (
        _mol_from_smiles,
        _get_pka_predictor,
    )

    predictor = _get_pka_predictor()

    acid_sites_col = df.get("acid_sites_json", pd.Series([None] * len(df), dtype=object)).values
    base_sites_col = df.get("base_sites_json", pd.Series([None] * len(df), dtype=object)).values

    t0 = time.time()
    n_done = sum(1 for x in acid_sites_col if x is not None and pd.notna(x))
    n_errors = 0

    for i, smiles in enumerate(df["Drug"]):
        # Skip already computed
        if acid_sites_col[i] is not None and pd.notna(acid_sites_col[i]):
            continue

        try:
            mol = _mol_from_smiles(smiles)
            prediction = predictor.predict(mol)
            acid_sites = prediction.acid_sites_1  # {atom_map: pKa}
            base_sites = prediction.base_sites_1

            # Convert keys to strings for JSON (atom map numbers are ints)
            acid_sites_col[i] = json.dumps({str(k): round(float(v), 4) for k, v in acid_sites.items()}) if acid_sites else "{}"
            base_sites_col[i] = json.dumps({str(k): round(float(v), 4) for k, v in base_sites.items()}) if base_sites else "{}"
        except Exception as e:
            acid_sites_col[i] = "{}"
            base_sites_col[i] = "{}"
            n_errors += 1

        n_done += 1
        if n_done % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(df) - n_done) / rate / 60 if rate > 0 else 0
            print(f"  {n_done}/{len(df)} ({n_errors} errors) — {rate:.1f} mol/s — ETA {eta:.1f} min")

    df["acid_sites_json"] = acid_sites_col
    df["base_sites_json"] = base_sites_col

    df.to_csv(CACHE_PATH, index=False)
    elapsed = time.time() - t0
    print(f"Done. {n_done} molecules in {elapsed/60:.1f} min ({n_errors} errors). Saved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
