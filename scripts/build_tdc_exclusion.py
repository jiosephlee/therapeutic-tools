"""Build TDC exclusion list: collect all val+test SMILES from TDC datasets,
canonicalize them, and save as JSON for ATTNSOM training exclusion."""

import os
import json
import csv
from rdkit import Chem

TDC_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "tdc", "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "ATTNSOM", "tdc_exclusion_smiles.json")


def main():
    all_smiles = set()
    tasks = sorted(d for d in os.listdir(TDC_RAW_DIR)
                   if os.path.isdir(os.path.join(TDC_RAW_DIR, d)))

    for task in tasks:
        task_dir = os.path.join(TDC_RAW_DIR, task)
        for split in ["val.csv", "test.csv"]:
            path = os.path.join(task_dir, split)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = row.get("Drug", "").strip()
                    if not smi:
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        all_smiles.add(Chem.MolToSmiles(mol))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sorted(all_smiles), f)

    print(f"Collected {len(all_smiles)} unique canonical SMILES from val+test across {len(tasks)} TDC tasks")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
