"""Convert ATTNSOM CYP SoM datasets to FAME3R format and train.

FAME3R expects SDF files with a <soms> field containing a Python list of
atom indices that are sites of metabolism. Our ATTNSOM datasets use
<PRIMARY_SOM>, <SECONDARY_SOM>, <TERTIARY_SOM> fields with single atom indices.

This script:
1. Merges all isoform SDF files, deduplicating by canonical SMILES
2. Converts SoM annotations to FAME3R's <soms> format
3. Writes a combined SDF for training
4. Runs `fame3r train`

Usage:
    python scripts/train_fame3r.py [--output-dir /path/to/models]
"""

import os
import sys
import argparse
import subprocess
from collections import defaultdict
from rdkit import Chem


CYP_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ATTNSOM", "cyp_dataset"
)
DEFAULT_OUTPUT_DIR = "/vast/projects/myatskar/design-documents/hf_home/fame3r_models"


def extract_som_indices(mol) -> set:
    """Extract all SoM atom indices from PRIMARY/SECONDARY/TERTIARY_SOM fields."""
    indices = set()
    for field in ["PRIMARY_SOM", "SECONDARY_SOM", "TERTIARY_SOM"]:
        if mol.HasProp(field):
            val = mol.GetProp(field).strip()
            if val and val not in ("", "N/A", "None", "-"):
                try:
                    # Could be single int or comma-separated
                    for part in val.split(","):
                        part = part.strip()
                        if part.isdigit():
                            indices.add(int(part))
                except (ValueError, AttributeError):
                    pass
    return indices


def build_merged_sdf(output_path: str):
    """Merge all CYP isoform SDF files into one FAME3R-compatible SDF."""
    sdf_files = sorted(
        f for f in os.listdir(CYP_DATASET_DIR) if f.endswith(".sdf")
    )

    # Deduplicate by canonical SMILES, merge SoM indices across isoforms
    mol_data = {}  # canonical_smiles -> (mol_block, som_indices, name)

    total_mols = 0
    for sdf_file in sdf_files:
        path = os.path.join(CYP_DATASET_DIR, sdf_file)
        isoform = sdf_file.replace(".sdf", "")
        suppl = Chem.SDMolSupplier(path, removeHs=False)

        for mol in suppl:
            if mol is None:
                continue
            total_mols += 1

            # Canonical SMILES for dedup
            mol_noH = Chem.RemoveHs(mol)
            canon = Chem.MolToSmiles(mol_noH)

            som_idx = extract_som_indices(mol)

            if canon in mol_data:
                # Merge SoM indices
                mol_data[canon]["soms"].update(som_idx)
                mol_data[canon]["isoforms"].append(isoform)
            else:
                mol_data[canon] = {
                    "mol": mol,
                    "soms": som_idx,
                    "name": mol.GetProp("_Name") if mol.HasProp("_Name") else canon,
                    "isoforms": [isoform],
                }

    print(f"Read {total_mols} molecules from {len(sdf_files)} SDF files")
    print(f"After deduplication: {len(mol_data)} unique molecules")

    # Count SoM statistics
    n_with_som = sum(1 for d in mol_data.values() if d["soms"])
    total_som_sites = sum(len(d["soms"]) for d in mol_data.values())
    print(f"Molecules with SoM annotations: {n_with_som}")
    print(f"Total SoM sites: {total_som_sites}")

    # Write FAME3R-compatible SDF
    # FAME3R uses CDPKit which is strict about UTF-8, so we rebuild
    # clean molecules with only the <soms> property
    writer = Chem.SDWriter(output_path)
    for canon, data in mol_data.items():
        orig_mol = data["mol"]
        # Create a fresh molecule copy to avoid inherited bad properties
        mol_block = Chem.MolToMolBlock(orig_mol)
        clean_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
        if clean_mol is None:
            continue
        # Sanitize name
        name = data["name"].encode("ascii", errors="replace").decode("ascii")
        clean_mol.SetProp("_Name", name)
        # Set the <soms> field as FAME3R expects
        soms_list = sorted(data["soms"])
        clean_mol.SetProp("soms", str(soms_list))
        writer.write(clean_mol)
    writer.close()

    print(f"Wrote {len(mol_data)} molecules to {output_path}")
    return len(mol_data)


def main():
    parser = argparse.ArgumentParser(description="Train FAME3R on CYP SoM data")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Model output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--sdf-only", action="store_true",
        help="Only generate the merged SDF, don't train"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sdf_path = os.path.join(args.output_dir, "fame3r_training_data.sdf")

    print("Step 1: Building merged SDF...")
    n_mols = build_merged_sdf(sdf_path)

    if args.sdf_only:
        print("Done (--sdf-only mode)")
        return

    print("\nStep 2: Training FAME3R...")
    model_dir = os.path.join(args.output_dir, "fame3r_model")
    cmd = [
        "fame3r", "train",
        "-i", sdf_path,
        "-o", model_dir,
        "--radius", "5",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"\nDone! Model saved to {model_dir}")
    print(f"To use: export FAME3R_MODEL_DIR={model_dir}")


if __name__ == "__main__":
    main()
