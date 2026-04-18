"""
Extend the fingerprints_with_canonicalized cache with any new canonical SMILES
introduced by data/tdc/deduplicated_no_salts.

For each task where both an existing npz and no-salts CSVs exist, this script:
  1. Loads the existing cache (fingerprints_with_canonicalized/{task}_embeddings.npz).
  2. Loads deduplicated_no_salts/{task}/{train,val,test}.csv (Drug column holds
     already-canonical salt-stripped SMILES).
  3. Appends any canonical SMILES not already present, computing morgan/feat_morgan
     fingerprints with the same generators used to build the original cache.
  4. Writes the combined cache to
     fingerprints_with_canonicalized_no_salts/{task}_embeddings.npz.

Splits column uses "train"/"val"/"test" tags from the no-salts CSVs for newly
added rows. Existing rows keep their original splits tag.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_fingerprint_embeddings import (
    _canonicalize_smiles,
    _compute_fingerprint,
    TASKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent
_REPO_ROOT_TT = _CACHE_DIR.parents[2]  # therapeutic-tuning repo root
_NO_SALTS_ROOT = _REPO_ROOT_TT / "data" / "tdc" / "deduplicated_no_salts"
_SRC_SUBDIR = "fingerprints_with_canonicalized"
_DST_SUBDIR = "fingerprints_with_canonicalized_no_salts"


def _load_no_salts(task: str) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for src_split in ("train", "val", "test"):
        path = _NO_SALTS_ROOT / task / f"{src_split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            smiles = row.get("Drug")
            label = row.get("Y")
            if pd.isna(smiles) or pd.isna(label):
                continue
            rows.append((str(smiles), int(label), src_split))
    return rows


def extend_task(task: str, overwrite: bool = False) -> bool:
    src_path = _CACHE_DIR / _SRC_SUBDIR / f"{task}_embeddings.npz"
    dst_dir = _CACHE_DIR / _DST_SUBDIR
    dst_path = dst_dir / f"{task}_embeddings.npz"

    if not src_path.exists():
        logger.warning("%s: source cache missing (%s) — skipping.", task, src_path)
        return False
    if dst_path.exists() and not overwrite:
        logger.info("%s: %s exists (use --overwrite to rebuild).", task, dst_path)
        return True

    src = np.load(src_path, allow_pickle=True)
    smiles = list(src["smiles"])
    canon = list(src["canonical_smiles"])
    morgan = list(src["morgan_fps"])
    feat = list(src["feat_morgan_fps"])
    labels = list(int(x) for x in src["labels"])
    splits = list(str(x) for x in src["splits"])

    existing_canon = set(canon)
    new_rows = _load_no_salts(task)
    if not new_rows:
        logger.warning("%s: no deduplicated_no_salts CSVs found — skipping.", task)
        return False

    added = 0
    fp_failed = 0
    canon_failed = 0
    for no_salt_smiles, label, split in new_rows:
        canonical = _canonicalize_smiles(no_salt_smiles)
        if canonical is None:
            canon_failed += 1
            continue
        if canonical in existing_canon:
            continue
        m = _compute_fingerprint(canonical, use_features=False)
        f = _compute_fingerprint(canonical, use_features=True)
        if m is None or f is None:
            fp_failed += 1
            continue
        smiles.append(no_salt_smiles)
        canon.append(canonical)
        morgan.append(m)
        feat.append(f)
        labels.append(int(label))
        splits.append(split)
        existing_canon.add(canonical)
        added += 1

    logger.info(
        "%s: existing=%d, added=%d (canon_fail=%d, fp_fail=%d) -> total=%d",
        task, len(src["smiles"]), added, canon_failed, fp_failed, len(smiles),
    )

    dst_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst_path,
        smiles=np.array(smiles, dtype=object),
        canonical_smiles=np.array(canon, dtype=object),
        morgan_fps=np.stack(morgan),
        feat_morgan_fps=np.stack(feat),
        labels=np.array(labels, dtype=np.int32),
        splits=np.array(splits, dtype=object),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks or TASKS
    rc = 0
    for task in tasks:
        try:
            ok = extend_task(task, overwrite=args.overwrite)
        except Exception as exc:
            logger.error("%s: failed: %s", task, exc)
            ok = False
            rc = 1
        if not ok:
            rc = max(rc, 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
