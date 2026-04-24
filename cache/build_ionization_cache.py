"""Build the ionization cache (Dimorphite-DL output) for all TDC SMILES.

Usage:
    python build_ionization_cache.py [--data_dir /path/to/TDC] [--workers 8] [--ph 7.4]

Output:
    ionization_cache.jsonl  — one JSON line per (SMILES, ph):
        {"smiles": "<canonical>", "ph": 7.4, "result": "<_compact_ionization output>"}
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd

EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}


def collect_all_smiles(data_dir: str) -> list:
    all_smiles = set()
    for dataset in sorted(os.listdir(data_dir)):
        if dataset in EXCLUDE_DATASETS:
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


def _canonicalize(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


_PH = 7.4


def compute_ionization(smiles: str) -> dict:
    from therapeutic_tools.adme import _compact_ionization

    canonical = _canonicalize(smiles)
    if canonical is None:
        return {"smiles": smiles, "ph": _PH, "result": None, "error": "invalid_smiles"}
    try:
        result = _compact_ionization(canonical, ph=_PH)
        return {"smiles": canonical, "ph": _PH, "result": result}
    except Exception as e:
        return {"smiles": canonical, "ph": _PH, "result": None, "error": str(e)}


def _init_worker(ph: float) -> None:
    global _PH
    _PH = ph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/data/TDC",
    )
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "ionization_cache.jsonl"),
    )
    args = parser.parse_args()

    print(f"Collecting SMILES from {args.data_dir}...")
    all_smiles = collect_all_smiles(args.data_dir)
    print(f"Found {len(all_smiles)} unique SMILES")

    done_keys = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done_keys.add((entry["smiles"], float(entry["ph"])))
                except Exception:
                    pass
        print(f"Resuming: {len(done_keys)} already cached")

    remaining = []
    for s in all_smiles:
        c = _canonicalize(s) or s
        if (c, args.ph) in done_keys:
            continue
        remaining.append(s)

    if not remaining:
        print("All SMILES already cached!")
        return

    start = time.time()
    completed = 0
    with open(args.output, "a") as out_f:
        with Pool(args.workers, initializer=_init_worker, initargs=(args.ph,)) as pool:
            for result in pool.imap_unordered(compute_ionization, remaining, chunksize=32):
                if result.get("result") is None:
                    continue
                out_f.write(json.dumps({
                    "smiles": result["smiles"],
                    "ph": result["ph"],
                    "result": result["result"],
                }) + "\n")
                completed += 1
                if completed % 2000 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate / 60
                    print(f"  {completed}/{len(remaining)} ({rate:.0f}/s, ETA: {eta:.1f}min)")

    elapsed = time.time() - start
    print(f"Done! {completed} molecules in {elapsed / 60:.1f} min")
    print(f"Cache saved to {args.output}")


if __name__ == "__main__":
    main()
