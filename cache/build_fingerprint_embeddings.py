"""
Build fingerprint-based embedding cache for therapeutic_tools KNN retrieval.

Computes Morgan and FeatureMorgan fingerprints directly from raw TDC CSVs
using RDKit.

Preprocessing:
  1. Reads from data/tdc/raw_deduplicated/ which has already been cleaned
     by scripts/deduplicate_tdc_train.py (conflicting labels dropped,
     same-label duplicates removed from train; val/test untouched).
  2. Stores the original dataset SMILES and a derived canonical SMILES for
     each entry. Fingerprints are computed from the canonical SMILES so
     equivalent string variants share the same representation.
  3. Morgan fingerprints: radius=2, nBits=2048
  4. FeatureMorgan fingerprints: radius=2, nBits=2048, useFeatures=True

Saves per-task npz files to:
    cache/<output_subdir>/{task}_embeddings.npz

Each npz contains:
    smiles          : (N,) object array of original dataset SMILES strings
    canonical_smiles: (N,) object array of canonical SMILES strings
    morgan_fps      : (N, 2048) uint8 fingerprint bit vectors
    feat_morgan_fps : (N, 2048) uint8 fingerprint bit vectors
    labels          : (N,) int32 binary class labels
    splits          : (N,) object array ("train" or "val")

Usage:
    python build_fingerprint_embeddings.py [--tasks TASK1 TASK2 ...] [--overwrite]
    python build_fingerprint_embeddings.py --output-subdir fingerprints_with_canonicalized
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent                    # .../therapeutic_tools/cache
_REPO_ROOT = _CACHE_DIR.parent.parent.parent.parent   # .../OpenRLHF-Tools
_RAW_DATA_DIR = _REPO_ROOT / "data" / "tdc" / "raw_deduplicated"
_RAW_FALLBACK_DATA_DIR = _REPO_ROOT / "data" / "tdc" / "raw"

TASKS = [
    "Carcinogens_Lagunin",
    "BBB_Martins",
    "DILI",
    "Pgp_Broccatelli",
    "PAMPA_NCATS",
    "HIA_Hou",
    "Bioavailability_Ma",
    "hERG",
    "AMES",
    "Skin_Reaction",
    "ClinTox",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
]

# Fingerprint parameters (matching Intern-S1 pipeline)
FP_RADIUS = 2
FP_NBITS = 2048


# ---------------------------------------------------------------------------
# Data loading & deduplication
# ---------------------------------------------------------------------------

def _load_split(task: str, split: str) -> pd.DataFrame:
    """Load a raw CSV split, returning DataFrame with Drug and Y columns."""
    candidates = [
        _RAW_DATA_DIR / task / f"{split}.csv",
        _RAW_FALLBACK_DATA_DIR / task / f"{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["Drug", "Y"])


def _canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string with RDKit."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _build_entries(task: str) -> tuple[list[tuple[str, str, int, str]], dict]:
    """Load (original_smiles, canonical_smiles, label, split) entries.

    Reads from raw_deduplicated/ where train has already been cleaned
    (no conflicting labels, no same-label duplicates).  Val is loaded as-is;
    val SMILES that duplicate a train SMILES are skipped.

    Returns:
        entries: list of (original_smiles, canonical_smiles, label, split) tuples
        stats: dict with counts
    """
    stats = {"train": 0, "val": 0, "val_overlap": 0, "canonicalization_failed": 0}

    entries = []
    seen = set()

    # Train (already deduplicated)
    train_df = _load_split(task, "train")
    for _, row in train_df.iterrows():
        smiles = row.get("Drug")
        label = row.get("Y")
        if pd.isna(smiles) or pd.isna(label):
            continue
        original_smi = str(smiles)
        if original_smi in seen:
            continue
        canonical_smi = _canonicalize_smiles(original_smi)
        if canonical_smi is None:
            stats["canonicalization_failed"] += 1
            continue
        entries.append((original_smi, canonical_smi, int(label), "train"))
        seen.add(original_smi)
        stats["train"] += 1

    # Val (keep all — even if SMILES overlaps with train, we need it as a query target;
    # neighbor restriction is handled by the splits field at query time)
    val_df = _load_split(task, "val")
    for _, row in val_df.iterrows():
        smiles = row.get("Drug")
        label = row.get("Y")
        if pd.isna(smiles) or pd.isna(label):
            continue
        original_smi = str(smiles)
        if original_smi in seen:
            stats["val_overlap"] += 1
            continue
        canonical_smi = _canonicalize_smiles(original_smi)
        if canonical_smi is None:
            stats["canonicalization_failed"] += 1
            continue
        entries.append((original_smi, canonical_smi, int(label), "val"))
        seen.add(original_smi)
        stats["val"] += 1

    return entries, stats


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def _make_generator(use_features: bool = False):
    """Create a Morgan fingerprint generator (new RDKit API, no deprecation warnings)."""
    from rdkit.Chem import rdFingerprintGenerator
    if use_features:
        inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    else:
        inv_gen = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_NBITS,
        atomInvariantsGenerator=inv_gen,
    )

_MORGAN_GEN = None
_FEAT_GEN = None

def _get_generators():
    global _MORGAN_GEN, _FEAT_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = _make_generator(use_features=False)
        _FEAT_GEN   = _make_generator(use_features=True)
    return _MORGAN_GEN, _FEAT_GEN


def _compute_fingerprint(smiles: str, use_features: bool = False) -> np.ndarray:
    """Compute Morgan fingerprint as uint8 array. Returns (2048,) or None."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen, feat_gen = _get_generators()
    gen = feat_gen if use_features else morgan_gen
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-task builder
# ---------------------------------------------------------------------------

def build_task(task: str, output_subdir: str = "fingerprint", overwrite: bool = False) -> bool:
    output_dir = _CACHE_DIR / output_subdir
    out_path = output_dir / f"{task}_embeddings.npz"
    if out_path.exists() and not overwrite:
        logger.info("%s: already exists, skipping (use --overwrite to rebuild).", task)
        return True

    # Check that raw data exists
    if not (_RAW_DATA_DIR / task / "train.csv").exists():
        logger.warning("%s: no raw train CSV found in %s, skipping.", task, _RAW_DATA_DIR / task)
        return False

    logger.info("%s: loading pre-deduplicated data...", task)
    entries, stats = _build_entries(task)
    if not entries:
        logger.warning("%s: no entries, skipping.", task)
        return False

    logger.info("%s: computing fingerprints for %d molecules...", task, len(entries))
    valid_smiles = []
    valid_canonical_smiles = []
    morgan_list = []
    feat_list = []
    labels_list = []
    splits_list = []
    n_failed = 0

    for original_smiles, canonical_smiles, label, split in entries:
        morgan = _compute_fingerprint(canonical_smiles, use_features=False)
        feat = _compute_fingerprint(canonical_smiles, use_features=True)
        if morgan is None or feat is None:
            n_failed += 1
            continue
        valid_smiles.append(original_smiles)
        valid_canonical_smiles.append(canonical_smiles)
        morgan_list.append(morgan)
        feat_list.append(feat)
        labels_list.append(label)
        splits_list.append(split)

    if stats["canonicalization_failed"]:
        logger.info("%s: %d SMILES failed canonicalization", task, stats["canonicalization_failed"])
    if n_failed:
        logger.info("%s: %d SMILES failed fingerprint computation", task, n_failed)

    if not valid_smiles:
        logger.warning("%s: no valid fingerprints computed, skipping.", task)
        return False

    smiles_arr = np.array(valid_smiles, dtype=object)
    canonical_smiles_arr = np.array(valid_canonical_smiles, dtype=object)
    morgan_arr = np.stack(morgan_list)
    feat_arr = np.stack(feat_list)
    labels_arr = np.array(labels_list, dtype=np.int32)
    splits_arr = np.array(splits_list, dtype=object)

    n_train = (splits_arr == "train").sum()
    n_val = (splits_arr == "val").sum()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        smiles=smiles_arr,
        canonical_smiles=canonical_smiles_arr,
        morgan_fps=morgan_arr,
        feat_morgan_fps=feat_arr,
        labels=labels_arr,
        splits=splits_arr,
    )
    logger.info(
        "%s: saved %d molecules (train=%d, val=%d) -> %s",
        task, len(valid_smiles), n_train, n_val, out_path,
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _RAW_DATA_DIR

    default_raw_data_dir = str(_RAW_DATA_DIR)
    parser = argparse.ArgumentParser(
        description="Build fingerprint embedding npz files from raw TDC CSVs."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific task names to build (default: all 16 TDC tasks).",
    )
    parser.add_argument(
        "--data-dir", default=default_raw_data_dir,
        help="Input TDC directory containing per-task split CSVs.",
    )
    parser.add_argument(
        "--output-subdir", default="fingerprint",
        help="Cache subdirectory under therapeutic_tools/cache/ for the output NPZ files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing npz files.",
    )
    args = parser.parse_args()

    _RAW_DATA_DIR = Path(args.data_dir)

    tasks = args.tasks if args.tasks else TASKS

    ok, failed = 0, 0
    for task in tasks:
        success = build_task(task, output_subdir=args.output_subdir, overwrite=args.overwrite)
        if success:
            ok += 1
        else:
            failed += 1

    logger.info("Done. %d succeeded, %d failed/skipped.", ok, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
