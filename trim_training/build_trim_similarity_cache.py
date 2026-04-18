"""
Phase 2: Build the per-task similarity cache that TRIM's CachedSimilarityRetriever needs.

Reads the fingerprint npz files already computed by build_fingerprint_embeddings.py
and produces per-query similarity pickle files in the layout:

    {trim_root}/data/cache/tdc_mol_fingerprints/
        Morgan_similarity/by_task/{task}/{split}_similarity.pkl
        Feature_Morgan_similarity/by_task/{task}/{split}_similarity.pkl

Each pickle is a dict:
    { query_smiles: {"label_0": [(score, ref_smiles), ...],
                     "label_1": [(score, ref_smiles), ...]} }

Usage:
    python build_trim_similarity_cache.py --trim-root /path/to/trim_artifacts
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CACHE_DIR = _REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
_DEFAULT_FP_SUBDIR = "fingerprints_with_canonicalized"
_DEFAULT_TRIM_ROOT = _REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "TRIM"

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "Carcinogens_Lagunin",
    "ClinTox", "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "DILI", "hERG", "HIA_Hou",
    "PAMPA_NCATS", "Pgp_Broccatelli", "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret", "Skin_Reaction",
]

TOP_K_PER_LABEL = 20  # how many neighbors per label to store


def tanimoto_matrix(query_fps: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto similarities between query and ref fingerprints.

    Args:
        query_fps: (Q, D) uint8 bit vectors
        ref_fps:   (R, D) uint8 bit vectors

    Returns:
        (Q, R) float32 similarity matrix
    """
    # Cast to float for safe arithmetic
    q = query_fps.astype(np.float32)
    r = ref_fps.astype(np.float32)
    # Intersection = dot product, Union = |a| + |b| - intersection
    intersection = q @ r.T
    q_bits = q.sum(axis=1, keepdims=True)  # (Q, 1)
    r_bits = r.sum(axis=1, keepdims=True)  # (R, 1)
    union = q_bits + r_bits.T - intersection  # (Q, R)
    # Avoid division by zero
    union = np.maximum(union, 1e-9)
    return intersection / union


def build_similarity_dict(
    query_smiles: np.ndarray,
    query_fps: np.ndarray,
    train_smiles: np.ndarray,
    train_fps: np.ndarray,
    train_labels: np.ndarray,
    top_k: int = TOP_K_PER_LABEL,
) -> dict[str, dict[str, list[tuple[float, str]]]]:
    """Build the similarity dict for one fingerprint type × one split."""
    n_query = len(query_smiles)
    n_train = len(train_smiles)
    log.info("  Computing %d × %d Tanimoto matrix ...", n_query, n_train)

    # Process in chunks to avoid memory issues on large tasks
    chunk_size = 500
    result: dict[str, dict[str, list[tuple[float, str]]]] = {}

    label_0_mask = train_labels == 0
    label_1_mask = train_labels == 1

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        chunk_fps = query_fps[start:end]
        sim_chunk = tanimoto_matrix(chunk_fps, train_fps)  # (chunk, n_train)

        for i, q_idx in enumerate(range(start, end)):
            q_smi = str(query_smiles[q_idx])
            scores = sim_chunk[i]

            # Top-K for label 0
            scores_0 = scores.copy()
            scores_0[~label_0_mask] = -1
            top_0_idx = np.argsort(scores_0)[::-1][:top_k]
            label_0_list = [
                (float(scores[idx]), str(train_smiles[idx]))
                for idx in top_0_idx
                if label_0_mask[idx]
            ]

            # Top-K for label 1
            scores_1 = scores.copy()
            scores_1[~label_1_mask] = -1
            top_1_idx = np.argsort(scores_1)[::-1][:top_k]
            label_1_list = [
                (float(scores[idx]), str(train_smiles[idx]))
                for idx in top_1_idx
                if label_1_mask[idx]
            ]

            result[q_smi] = {"label_0": label_0_list, "label_1": label_1_list}

    return result


def build_task_cache(
    task: str,
    trim_root: Path,
    fp_subdir: str = _DEFAULT_FP_SUBDIR,
    overwrite: bool = False,
) -> bool:
    """Build Morgan + FeatureMorgan similarity caches for one task."""
    npz_path = _CACHE_DIR / fp_subdir / f"{task}_embeddings.npz"
    if not npz_path.exists():
        log.warning("Skipping %s: no embeddings at %s", task, npz_path)
        return False

    cache_root = trim_root / "data" / "cache" / "tdc_mol_fingerprints"

    # Check if already done
    morgan_train = cache_root / "Morgan_similarity" / "by_task" / task / "train_similarity.pkl"
    feat_train = cache_root / "Feature_Morgan_similarity" / "by_task" / task / "train_similarity.pkl"
    if morgan_train.exists() and feat_train.exists() and not overwrite:
        log.info("%-35s already exists, skipping", task)
        return True

    log.info("%-35s loading embeddings ...", task)
    data = np.load(npz_path, allow_pickle=True)
    smiles = data["smiles"]
    morgan_fps = data["morgan_fps"]
    feat_morgan_fps = data["feat_morgan_fps"]
    labels = data["labels"]
    splits = data["splits"]

    train_mask = splits == "train"
    val_mask = splits == "val"

    train_smiles = smiles[train_mask]
    train_morgan = morgan_fps[train_mask]
    train_feat = feat_morgan_fps[train_mask]
    train_labels = labels[train_mask]

    # For each split and each fingerprint type, build similarity dict
    for fp_name, train_fps_all, query_fps_dict in [
        ("Morgan_similarity", train_morgan, {"train": train_morgan, "valid": morgan_fps[val_mask]}),
        ("Feature_Morgan_similarity", train_feat, {"train": train_feat, "valid": feat_morgan_fps[val_mask]}),
    ]:
        for split_name, query_fps in query_fps_dict.items():
            if split_name == "train":
                query_smi = train_smiles
            else:
                query_smi = smiles[val_mask]

            if len(query_smi) == 0:
                log.warning("  %s/%s/%s: no queries, writing empty dict", task, fp_name, split_name)
                sim_dict: dict = {}
            else:
                log.info("  %s %s/%s (%d queries, %d refs) ...",
                         task, fp_name, split_name, len(query_smi), len(train_smiles))
                sim_dict = build_similarity_dict(
                    query_smi, query_fps, train_smiles, train_fps_all, train_labels,
                )

            out_path = cache_root / fp_name / "by_task" / task / f"{split_name}_similarity.pkl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                pickle.dump(sim_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info("  Wrote %s (%d entries)", out_path.name, len(sim_dict))

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TRIM similarity cache from fingerprint embeddings.")
    parser.add_argument("--trim-root", type=str, default=str(_DEFAULT_TRIM_ROOT))
    parser.add_argument("--fp-subdir", type=str, default=_DEFAULT_FP_SUBDIR,
                        help="Subdirectory under cache/ containing the npz files")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trim_root = Path(args.trim_root).resolve()
    tasks = args.tasks or TASKS

    ok, fail = 0, 0
    for task in tasks:
        if build_task_cache(task, trim_root, fp_subdir=args.fp_subdir, overwrite=args.overwrite):
            ok += 1
        else:
            fail += 1

    log.info("Done: %d succeeded, %d failed/skipped.", ok, fail)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
