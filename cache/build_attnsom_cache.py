"""
Precompute ATTNSOM SoM predictions for all TDC SMILES and save as JSONL cache.

Usage:
    python build_attnsom_cache.py [--checkpoint /path/to/attnsom_checkpoint.pt]

Output:
    attnsom_cache.jsonl — one JSON line per SMILES: {"smiles": "...", "result": "..."}
    where "result" is the formatted prediction string returned by the tool.
"""

import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
from rdkit import Chem


EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}

DEFAULT_CHECKPOINT = (
    "/vast/projects/myatskar/design-documents/hf_home/attnsom_results/attnsom_checkpoint.pt"
)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "tdc", "raw"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default=os.path.abspath(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "attnsom_cache.jsonl"),
    )
    args = parser.parse_args()

    print(f"Loading ATTNSOM from {args.checkpoint}")
    from ATTNSOM.inference import ATTNSOMPredictor, format_prediction

    predictor = ATTNSOMPredictor(args.checkpoint)

    print(f"Collecting SMILES from {args.data_dir}...")
    all_smiles = collect_all_smiles(args.data_dir)
    print(f"Found {len(all_smiles)} unique SMILES")

    # Resume support
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
    errors = 0
    with open(args.output, "a") as out_f:
        for smi in remaining:
            try:
                # Canonicalize to match what the tool will look up
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    errors += 1
                    continue
                canon = Chem.MolToSmiles(mol)
                result = predictor.predict(canon)
                formatted = format_prediction(result)
                out_f.write(json.dumps({"smiles": canon, "result": formatted}) + "\n")
            except Exception as e:
                out_f.write(json.dumps({"smiles": smi, "result": f"Error: {e}"}) + "\n")
                errors += 1

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
    print(f"Done! {completed} molecules in {elapsed:.1f}s ({errors} errors)")
    print(f"Cache saved to {args.output}")


if __name__ == "__main__":
    main()
