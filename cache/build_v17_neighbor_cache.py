"""
Build the v17 per-molecule neighbor-tool cache for all TDC SMILES.

One JSON line per canonical SMILES with precomputed artifacts:
  - murcko_scaffold
  - functional_groups (sorted list)
  - size_shape (MW / heavy atoms / ring_total / fraction_csp3 / rotatable_bonds)
  - mmp_single (single-cut fragmentation map)
  - mmp_double (double-cut fragmentation map, with both label orderings)

Usage:
    python build_v17_neighbor_cache.py [--data_dir /path/to/TDC] [--workers 8]
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pandas as pd


EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}
CACHE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = os.environ.get(
    "THERAPEUTIC_TOOLS_TDC_DATA_DIR",
    str(CACHE_DIR / "tdc" / "raw"),
)


def collect_all_smiles(data_dir: str) -> list:
    all_smiles = set()
    for dataset in sorted(os.listdir(data_dir)):
        if dataset in EXCLUDE_DATASETS:
            print(f"  Skipping excluded dataset: {dataset}")
            continue
        dpath = os.path.join(data_dir, dataset)
        if not os.path.isdir(dpath):
            continue
        for split in ["train.csv", "val.csv", "test.csv"]:
            fpath = os.path.join(dpath, split)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                if "Drug" in df.columns:
                    all_smiles.update(df["Drug"].dropna().unique())
    return sorted(all_smiles)


def _compute_entry(smiles: str) -> dict:
    try:
        from therapeutic_tools.tools.v17_cache import compute_artifacts

        return compute_artifacts(smiles)
    except Exception as e:
        return {"smiles": smiles, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "v17_neighbor_cache.jsonl"),
    )
    args = parser.parse_args()

    print(f"Collecting SMILES from {args.data_dir}...")
    all_smiles = collect_all_smiles(args.data_dir)
    print(f"Found {len(all_smiles)} unique SMILES")

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry.get("smiles"))
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already cached, {len(all_smiles) - len(done)} remaining")
    remaining = [s for s in all_smiles if s not in done]

    if not remaining:
        print("All SMILES already cached!")
        return

    start = time.time()
    completed = 0
    with open(args.output, "a") as out_f:
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(_compute_entry, remaining, chunksize=32):
                out_f.write(json.dumps(result) + "\n")
                completed += 1
                if completed % 2000 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate / 60
                    print(
                        f"  {completed}/{len(remaining)} "
                        f"({rate:.1f}/s, ETA: {eta:.1f}min)"
                    )

    elapsed = time.time() - start
    print(f"Done! {completed} molecules in {elapsed / 60:.1f} min")
    print(f"Cache saved to {args.output}")


if __name__ == "__main__":
    main()
