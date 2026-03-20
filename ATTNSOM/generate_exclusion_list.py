"""
Generate a canonical SMILES exclusion list from all TDC val+test splits.

These SMILES must be excluded from ATTNSOM training to prevent data leakage
when ATTNSOM predictions are used as tools during TDC evaluation.
"""
import csv
import os
import glob
import json

from rdkit import Chem


def collect_tdc_eval_smiles(tdc_dir: str) -> set:
    """Collect all canonical SMILES from TDC val and test CSVs."""
    smiles_set = set()
    for split in ["val.csv", "test.csv"]:
        for fpath in glob.glob(os.path.join(tdc_dir, "*", split)):
            with open(fpath) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    smi = row.get("Drug", "")
                    if smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            smiles_set.add(Chem.MolToSmiles(mol))
    return smiles_set


if __name__ == "__main__":
    tdc_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "data", "tdc", "raw"
    )
    tdc_dir = os.path.abspath(tdc_dir)
    print(f"TDC raw dir: {tdc_dir}")

    smiles_set = collect_tdc_eval_smiles(tdc_dir)
    print(f"Collected {len(smiles_set)} unique canonical SMILES from TDC val+test")

    out_path = os.path.join(os.path.dirname(__file__), "tdc_exclusion_smiles.json")
    with open(out_path, "w") as f:
        json.dump(sorted(smiles_set), f)
    print(f"Saved to {out_path}")
