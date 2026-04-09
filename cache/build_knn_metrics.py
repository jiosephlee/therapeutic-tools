"""
Precompute KNN accuracy and F1 per TDC task for both embedding types.

Supports two embedding types:
  learned     — neural/AttNSOM embeddings, cosine similarity (default)
  fingerprint — Morgan + FeatureMorgan bit vectors, weighted Tanimoto

Cache layout:
  cache/learned/{task}_embeddings.npz      — smiles, embeddings, labels
  cache/<fingerprint_subdir>/{task}_embeddings.npz
      — smiles, canonical_smiles, morgan_fps, feat_morgan_fps, labels

Saves per-task metrics (val_accuracy, val_f1, test_accuracy, test_f1) to
knn_metrics.json with structure:
  {
    "learned":     {task: {val_accuracy, val_f1, ...}},
    "fingerprint": {task: {val_accuracy, val_f1, ...}}
  }

Usage:
    python build_knn_metrics.py [--k 27] [--embedding-type learned|fingerprint|both]
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(CACHE_DIR, "..", "..", "..", "..")
_DATA_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "data", "tdc", "raw"),
    os.path.join(_PROJECT_ROOT, "data", "TDC"),
]
EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}
FINGERPRINT_SUBDIR_CANDIDATES = [
    os.environ.get("THERAPEUTIC_FINGERPRINT_CACHE_SUBDIR"),
    "fingerprints_with_canonicalized",
    "fingerprint_v8",
    "fingerprint",
]


def _find_data_dir():
    for d in _DATA_CANDIDATES:
        if os.path.isdir(d):
            return d
    return _DATA_CANDIDATES[0]


def _resolve_fingerprint_dir(preferred_subdir: str | None = None) -> str:
    candidates = [preferred_subdir] if preferred_subdir else []
    candidates.extend(FINGERPRINT_SUBDIR_CANDIDATES)
    for subdir in candidates:
        if not subdir:
            continue
        path = os.path.join(CACHE_DIR, subdir)
        if os.path.isdir(path):
            return path
    return os.path.join(CACHE_DIR, "fingerprint")


# ---------------------------------------------------------------------------
# Learned embeddings: cosine KNN
# ---------------------------------------------------------------------------

def cosine_knn_predict(query_emb, train_embs, train_labels, k, exclude_idx=None):
    """Return (predicted_label, neighbor_labels, neighbor_sims)."""
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8
    sims = (train_embs / norms) @ q_norm

    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    top_k = np.argsort(sims)[::-1][:k]
    neighbor_labels = train_labels[top_k].astype(int)
    pred = int(np.round(neighbor_labels.mean()))
    return pred, neighbor_labels, sims[top_k]


# ---------------------------------------------------------------------------
# Fingerprint embeddings: weighted Tanimoto KNN
# ---------------------------------------------------------------------------

def _tanimoto_bulk(query_fp: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """Batch Tanimoto of query_fp (1D) against ref_fps (N x D)."""
    a_and_b = ref_fps @ query_fp
    a_bits = query_fp.sum()
    b_bits = ref_fps.sum(axis=1)
    denom = a_bits + b_bits - a_and_b
    return np.where(denom > 0, a_and_b / denom, 0.0)


def fingerprint_knn_predict(
    query_morgan, query_feat,
    train_morgans, train_feats, train_labels,
    k, exclude_idx=None,
    w_morgan=0.8, w_feat=0.2,
):
    """Weighted Tanimoto KNN prediction for fingerprint embeddings."""
    sims = w_morgan * _tanimoto_bulk(query_morgan, train_morgans) \
         + w_feat   * _tanimoto_bulk(query_feat,   train_feats)

    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    top_k = np.argsort(sims)[::-1][:k]
    neighbor_labels = train_labels[top_k].astype(int)
    pred = int(np.round(neighbor_labels.mean()))
    return pred, neighbor_labels, sims[top_k]


# ---------------------------------------------------------------------------
# Per-task evaluators
# ---------------------------------------------------------------------------

def evaluate_task_learned(task, k=27):
    """Compute KNN metrics for a single task using learned embeddings."""
    npz_path = os.path.join(CACHE_DIR, "learned", f"{task}_embeddings.npz")
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path, allow_pickle=True)
    all_smiles = d["smiles"]
    all_embs   = d["embeddings"].astype(np.float32)
    all_labels = d["labels"]

    data_dir = os.path.abspath(_find_data_dir())
    train_path = os.path.join(data_dir, task, "train.csv")
    val_path   = os.path.join(data_dir, task, "val.csv")
    test_path  = os.path.join(data_dir, task, "test.csv")

    if not os.path.exists(train_path):
        return None

    train_smi = set(pd.read_csv(train_path)["Drug"].dropna())
    val_smi   = set(pd.read_csv(val_path)["Drug"].dropna())   if os.path.exists(val_path)  else set()
    test_smi  = set(pd.read_csv(test_path)["Drug"].dropna())  if os.path.exists(test_path) else set()

    smi_list  = list(all_smiles)
    train_idx = [i for i, s in enumerate(smi_list) if s in train_smi]
    val_idx   = [i for i, s in enumerate(smi_list) if s in val_smi]
    test_idx  = [i for i, s in enumerate(smi_list) if s in test_smi]

    if not train_idx or (not val_idx and not test_idx):
        return None

    train_embs   = all_embs[train_idx]
    train_labels = all_labels[train_idx]

    def _predict_split(split_idx):
        y_true, y_pred = [], []
        for idx in split_idx:
            emb = all_embs[idx]
            label = int(all_labels[idx])
            exclude = train_idx.index(idx) if idx in train_idx else None
            pred, _, _ = cosine_knn_predict(emb, train_embs, train_labels, k, exclude_idx=exclude)
            y_true.append(label)
            y_pred.append(pred)
        if not y_true:
            return None, None
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro', zero_division=0)

    train_acc, train_f1 = _predict_split(train_idx)
    val_acc, val_f1     = _predict_split(val_idx)
    test_acc, test_f1   = _predict_split(test_idx)

    result = {"task": task, "k": k, "n_train": len(train_idx)}
    if train_acc is not None: result.update({"train_accuracy": round(train_acc, 4), "train_f1": round(train_f1, 4)})
    if val_acc   is not None: result.update({"val_accuracy":   round(val_acc,   4), "val_f1":   round(val_f1,   4), "n_val":  len(val_idx)})
    if test_acc  is not None: result.update({"test_accuracy":  round(test_acc,  4), "test_f1":  round(test_f1,  4), "n_test": len(test_idx)})
    return result


def evaluate_task_fingerprint(task, k=27, fingerprint_dir: str | None = None):
    """Compute weighted-Tanimoto KNN metrics for a single task using fingerprint embeddings."""
    npz_path = os.path.join(_resolve_fingerprint_dir(fingerprint_dir), f"{task}_embeddings.npz")
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path, allow_pickle=True)
    all_smiles  = d["smiles"]
    all_morgans = d["morgan_fps"].astype(np.float32)
    all_feats   = d["feat_morgan_fps"].astype(np.float32)
    all_labels  = d["labels"]

    # Use split membership stored in the npz (canonical SMILES don't match raw CSVs)
    if "splits" in d:
        all_splits = d["splits"]
    else:
        # Fallback: try matching against raw CSVs (works only for non-canonical SMILES)
        data_dir = os.path.abspath(_find_data_dir())
        train_smi = set(pd.read_csv(os.path.join(data_dir, task, "train.csv"))["Drug"].dropna())
        val_smi   = set(pd.read_csv(os.path.join(data_dir, task, "val.csv"))["Drug"].dropna()) if os.path.exists(os.path.join(data_dir, task, "val.csv")) else set()
        all_splits = np.array(["train" if s in train_smi else ("val" if s in val_smi else "other") for s in all_smiles])

    train_idx = [i for i, sp in enumerate(all_splits) if sp == "train"]
    val_idx   = [i for i, sp in enumerate(all_splits) if sp == "val"]
    test_idx  = [i for i, sp in enumerate(all_splits) if sp == "test"]

    if not train_idx or (not val_idx and not test_idx):
        return None

    train_idx_set = set(train_idx)
    train_morgans = all_morgans[train_idx]
    train_feats   = all_feats[train_idx]
    train_labels  = all_labels[train_idx]

    def _predict_split(split_idx):
        y_true, y_pred = [], []
        for idx in split_idx:
            q_morgan = all_morgans[idx]
            q_feat   = all_feats[idx]
            label    = int(all_labels[idx])
            exclude  = train_idx.index(idx) if idx in train_idx_set else None
            pred, _, _ = fingerprint_knn_predict(
                q_morgan, q_feat,
                train_morgans, train_feats, train_labels,
                k, exclude_idx=exclude,
            )
            y_true.append(label)
            y_pred.append(pred)
        if not y_true:
            return None, None
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro', zero_division=0)

    train_acc, train_f1 = _predict_split(train_idx)
    val_acc, val_f1     = _predict_split(val_idx)
    test_acc, test_f1   = _predict_split(test_idx)

    result = {"task": task, "k": k, "n_train": len(train_idx)}
    if train_acc is not None: result.update({"train_accuracy": round(train_acc, 4), "train_f1": round(train_f1, 4)})
    if val_acc   is not None: result.update({"val_accuracy":   round(val_acc,   4), "val_f1":   round(val_f1,   4), "n_val":  len(val_idx)})
    if test_acc  is not None: result.update({"test_accuracy":  round(test_acc,  4), "test_f1":  round(test_f1,  4), "n_test": len(test_idx)})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=27)
    parser.add_argument(
        "--embedding-type",
        choices=["learned", "fingerprint", "both"],
        default="both",
        help="Which embedding type(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--fingerprint-subdir",
        default=None,
        help="Fingerprint cache subdirectory under therapeutic_tools/cache/. Defaults to auto-detect.",
    )
    args = parser.parse_args()

    embedding_types = ["learned", "fingerprint"] if args.embedding_type == "both" else [args.embedding_type]

    # Load existing metrics to merge into (preserve unchanged types)
    out_path = os.path.join(CACHE_DIR, "knn_metrics.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        # Upgrade flat format to nested if needed
        if not any(k in all_results for k in ("learned", "fingerprint")):
            all_results = {"learned": all_results}
    else:
        all_results = {}

    for emb_type in embedding_types:
        type_dir = _resolve_fingerprint_dir(args.fingerprint_subdir) if emb_type == "fingerprint" else os.path.join(CACHE_DIR, emb_type)
        tasks = []
        if os.path.isdir(type_dir):
            for fname in sorted(os.listdir(type_dir)):
                if fname.endswith("_embeddings.npz"):
                    task = fname[: -len("_embeddings.npz")]
                    tasks.append(task)

        print(f"\n--- {emb_type}: {len(tasks)} tasks ---")
        type_results = {}
        for task in tasks:
            if task in EXCLUDE_DATASETS:
                continue
            print(f"  {task}...", end=" ", flush=True)
            if emb_type == "learned":
                metrics = evaluate_task_learned(task, k=args.k)
            else:
                metrics = evaluate_task_fingerprint(task, k=args.k, fingerprint_dir=args.fingerprint_subdir)
            if metrics:
                type_results[task] = metrics
                parts = []
                if "train_accuracy" in metrics:
                    parts.append(f"train_acc={metrics['train_accuracy']:.4f}, train_f1={metrics['train_f1']:.4f}")
                if "val_accuracy" in metrics:
                    parts.append(f"val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")
                if "test_accuracy" in metrics:
                    parts.append(f"test_acc={metrics['test_accuracy']:.4f}, test_f1={metrics['test_f1']:.4f}")
                print(", ".join(parts))
            else:
                print("skipped")

        all_results[emb_type] = type_results

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
