"""
Phase 1: Prepare all data artifacts that TRIM training needs.

Converts data from OpenRLHF-Tools formats into the directory layout
expected by the TRIM training pipeline:

  1. Processed JSONL splits  (CSV → JSONL, "Drug" → "drug", "val" → "valid")
  2. RDKit + pKa feature CSV  (from tdc_metadata_consolidated.csv)
  3. Functional-group top-level binary CSV  (from fg_cache.jsonl)

Usage:
    python prepare_trim_data.py --trim-root /vast/projects/.../trim_artifacts
    python prepare_trim_data.py --trim-root ./TRIM   # default = TRIM submodule
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths relative to this repo
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]  # OpenRLHF-Tools
_RAW_DEDUP_DIR = _REPO_ROOT / "data" / "tdc" / "raw_deduplicated"
_CACHE_DIR = _REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
_METADATA_CSV = _CACHE_DIR / "tdc_metadata_consolidated.csv"
_FG_CACHE_JSONL = _CACHE_DIR / "fg_cache.jsonl"
_DEFAULT_TRIM_ROOT = _REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "TRIM"

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "Carcinogens_Lagunin",
    "ClinTox", "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "DILI", "hERG", "HIA_Hou",
    "PAMPA_NCATS", "Pgp_Broccatelli", "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret", "Skin_Reaction",
]

# Columns from tdc_metadata_consolidated.csv to keep for the RDKit+pKa
# feature CSV.  Matches the "core_pka_no_fr_counts" feature set name:
# all numeric RDKit descriptors + core pKa columns, NO fragment-count (fr_*)
# columns, NO JSON blob columns.
_DROP_COLUMNS = {
    "Drug",             # SMILES key — written separately as "smiles"
    "acid_sites_json",  # JSON, not numeric
    "base_sites_json",  # JSON, not numeric
}


# ===================================================================
# Step 1 — Processed TDC JSONL splits
# ===================================================================

def convert_csv_to_jsonl(
    csv_path: Path,
    jsonl_path: Path,
    smiles_col: str = "Drug",
    label_col: str = "Y",
) -> int:
    """Convert a raw_deduplicated CSV to TRIM's JSONL format."""
    df = pd.read_csv(csv_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            smiles = row.get(smiles_col)
            label = row.get(label_col)
            if pd.isna(smiles) or pd.isna(label):
                continue
            record = {"drug": str(smiles), "Y": int(label)}
            out.write(json.dumps(record) + "\n")
            count += 1
    return count


def build_jsonl_splits(trim_root: Path) -> None:
    """Convert all tasks × {train,valid} from raw_deduplicated CSVs."""
    processed_root = trim_root / "data" / "processed" / "tdc_no_conflict_labels_salt_removed"
    for task in TASKS:
        # Train
        train_csv = _RAW_DEDUP_DIR / task / "train.csv"
        if not train_csv.exists():
            log.warning("Skipping %s: no train.csv at %s", task, train_csv)
            continue
        train_jsonl = processed_root / "train" / f"{task}.jsonl"
        n_train = convert_csv_to_jsonl(train_csv, train_jsonl)

        # Valid (raw_deduplicated calls it "val", TRIM calls it "valid")
        val_csv = _RAW_DEDUP_DIR / task / "val.csv"
        valid_jsonl = processed_root / "valid" / f"{task}.jsonl"
        n_valid = 0
        if val_csv.exists():
            n_valid = convert_csv_to_jsonl(val_csv, valid_jsonl)
        else:
            log.warning("No val.csv for %s — creating empty valid split", task)
            valid_jsonl.parent.mkdir(parents=True, exist_ok=True)
            valid_jsonl.write_text("")

        log.info("%-35s train=%d  valid=%d", task, n_train, n_valid)


# ===================================================================
# Step 2 — RDKit + pKa feature CSV
# ===================================================================

def build_rdkit_feature_csv(trim_root: Path) -> None:
    """Reshape tdc_metadata_consolidated.csv into the layout TRIM expects.

    Output goes to BOTH feature-set paths referenced by TRIM configs so that
    either the full or core_pka_no_fr_counts config just works.
    """
    log.info("Loading %s ...", _METADATA_CSV)
    df = pd.read_csv(_METADATA_CSV, low_memory=False)

    # Rename SMILES column
    if "Drug" not in df.columns:
        raise KeyError(f"Expected 'Drug' column in {_METADATA_CSV}")
    df = df.rename(columns={"Drug": "smiles"})

    # Drop non-numeric / non-feature columns
    cols_to_drop = [c for c in _DROP_COLUMNS if c in df.columns and c != "Drug"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Deduplicate by SMILES (keep first)
    before = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first")
    log.info("Deduped by SMILES: %d → %d rows", before, len(df))

    # Write to the two paths TRIM configs reference
    targets = [
        trim_root / "data" / "features"
        / "rdkit_descriptors_and_pka_easy_to_NLP_Lv1"
        / "tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka_easy_to_NLP_Lv1.csv",
        trim_root / "data" / "features"
        / "rdkit_descriptors_and_pka_easy_to_NLP_Lv1_core_pka_no_fr_counts"
        / "tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka_easy_to_NLP_Lv1_core_pka_no_fr_counts.csv",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
        log.info("Wrote rdkit feature CSV: %s  (%d rows × %d cols)", target.name, len(df), len(df.columns))


# ===================================================================
# Step 3 — Functional-group top-level binary CSV
# ===================================================================

def build_fg_csv(trim_root: Path) -> None:
    """Convert fg_cache.jsonl (text descriptions) to a binary FG vector CSV."""
    log.info("Loading %s ...", _FG_CACHE_JSONL)

    # First pass: discover all unique FG names
    fg_universe: set[str] = set()
    entries: list[tuple[str, list[str]]] = []

    with _FG_CACHE_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            smiles = record["smiles"]
            fg_text = record["fg"]
            fg_names: list[str] = []
            for part in fg_text.split("\n"):
                part = part.strip()
                if part.startswith("- "):
                    name = part.split(":")[0][2:].strip()
                    # Strip multiplicity markers like " (x2)"
                    name = re.sub(r"\s*\(x\d+\)\s*$", "", name)
                    fg_names.append(name)
                    fg_universe.add(name)
            entries.append((smiles, fg_names))

    sorted_fgs = sorted(fg_universe)
    log.info("FG universe: %d unique groups across %d molecules", len(sorted_fgs), len(entries))

    # Build binary matrix
    rows: list[dict[str, object]] = []
    for smiles, fg_names in entries:
        row: dict[str, object] = {"smiles": smiles}
        present = set(fg_names)
        for fg in sorted_fgs:
            row[fg] = 1 if fg in present else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    # Deduplicate by SMILES
    before = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first")
    log.info("Deduped FG CSV: %d → %d rows", before, len(df))

    target = (
        trim_root / "data" / "features" / "fg_top_level"
        / "tdc_no_conflict_labels_salt_removed_unique_smiles_top_level_fg_vectors.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    log.info("Wrote FG CSV: %s  (%d rows × %d cols)", target.name, len(df), len(df.columns))


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TRIM data artifacts from OpenRLHF-Tools data.")
    parser.add_argument(
        "--trim-root",
        type=str,
        default=str(_DEFAULT_TRIM_ROOT),
        help="Root directory for TRIM data/feature/output artifacts.  "
             "This becomes what TRIM_PROJECT_ROOT should point to.  "
             f"Default: {_DEFAULT_TRIM_ROOT}",
    )
    parser.add_argument("--skip-jsonl", action="store_true", help="Skip JSONL conversion")
    parser.add_argument("--skip-rdkit", action="store_true", help="Skip RDKit feature CSV")
    parser.add_argument("--skip-fg", action="store_true", help="Skip FG binary CSV")
    args = parser.parse_args()

    trim_root = Path(args.trim_root).resolve()
    log.info("TRIM root: %s", trim_root)

    if not args.skip_jsonl:
        log.info("=== Step 1: Building JSONL splits ===")
        build_jsonl_splits(trim_root)

    if not args.skip_rdkit:
        log.info("=== Step 2: Building RDKit + pKa feature CSV ===")
        build_rdkit_feature_csv(trim_root)

    if not args.skip_fg:
        log.info("=== Step 3: Building FG top-level binary CSV ===")
        build_fg_csv(trim_root)

    log.info("Done. Set TRIM_PROJECT_ROOT=%s to use these artifacts.", trim_root)


if __name__ == "__main__":
    main()
