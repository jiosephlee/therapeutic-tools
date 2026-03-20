"""
Build the concise functional group cache for all TDC SMILES.

Usage:
    python build_fg_cache.py [--data_dir /path/to/TDC] [--workers 8]

Output:
    fg_cache.jsonl  — one JSON line per SMILES: {"smiles": "...", "fg": "..."}
"""

import os
import sys
import json
import argparse
import time
from multiprocessing import Pool, cpu_count

# Add parent dirs so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd


EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}


def collect_all_smiles(data_dir: str) -> list:
    """Collect all unique SMILES from TDC train/val/test splits."""
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


def compute_fg(smiles: str) -> dict:
    """Compute concise FG description for a single SMILES."""
    from legacy_tools.AccFG import concise_fg_description
    try:
        desc = concise_fg_description(smiles)
        return {"smiles": smiles, "fg": desc}
    except Exception as e:
        return {"smiles": smiles, "fg": f"Error: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/data/TDC",
    )
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "fg_cache.jsonl"),
    )
    args = parser.parse_args()

    print(f"Collecting SMILES from {args.data_dir}...")
    all_smiles = collect_all_smiles(args.data_dir)
    print(f"Found {len(all_smiles)} unique SMILES")

    # Skip already-cached entries if output file exists (resume support)
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry["smiles"])
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
            for result in pool.imap_unordered(compute_fg, remaining, chunksize=64):
                out_f.write(json.dumps(result) + "\n")
                completed += 1
                if completed % 5000 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate / 60
                    print(
                        f"  {completed}/{len(remaining)} "
                        f"({rate:.0f}/s, ETA: {eta:.1f}min)"
                    )

    elapsed = time.time() - start
    print(f"Done! {completed} molecules in {elapsed / 60:.1f} min")
    print(f"Cache saved to {args.output}")


if __name__ == "__main__":
    main()
