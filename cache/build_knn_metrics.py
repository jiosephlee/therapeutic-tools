"""
Precompute KNN accuracy and F1 per TDC task using cached embeddings.

For each task, evaluates KNN (k=27) on the val and test splits separately.
Saves per-task metrics (val_accuracy, val_f1, test_accuracy, test_f1) to
knn_metrics.json.

Usage:
    python build_knn_metrics.py [--k 27]
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

CACHE_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.join(CACHE_DIR, "..", "..", "..")
_DATA_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, "data", "tdc", "raw"),
    os.path.join(_PROJECT_ROOT, "data", "TDC"),
]
EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}


def _find_data_dir():
    for d in _DATA_CANDIDATES:
        if os.path.isdir(d):
            return d
    return _DATA_CANDIDATES[0]  # fallback


def cosine_knn_predict(query_emb, train_embs, train_labels, k, exclude_idx=None):
    """Return (predicted_label, neighbor_labels, neighbor_sims)."""
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8
    sims = (train_embs / norms) @ q_norm

    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    top_k = np.argsort(sims)[::-1][:k]
    neighbor_labels = train_labels[top_k].astype(int)
    neighbor_sims = sims[top_k]

    # Majority vote
    pred = int(np.round(neighbor_labels.mean()))
    return pred, neighbor_labels, neighbor_sims


def evaluate_task(task, k=27):
    """Compute KNN metrics for a single task."""
    npz_path = os.path.join(CACHE_DIR, task, "embeddings.npz")
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path, allow_pickle=True)
    all_smiles = d["smiles"]
    all_embs = d["embeddings"].astype(np.float32)
    all_labels = d["labels"]

    # Load train/val/test splits
    data_dir = os.path.abspath(_find_data_dir())
    train_path = os.path.join(data_dir, task, "train.csv")
    val_path = os.path.join(data_dir, task, "val.csv")
    test_path = os.path.join(data_dir, task, "test.csv")

    if not os.path.exists(train_path):
        return None

    train_smi = set(pd.read_csv(train_path)["Drug"].dropna())
    val_smi = set(pd.read_csv(val_path)["Drug"].dropna()) if os.path.exists(val_path) else set()
    test_smi = set(pd.read_csv(test_path)["Drug"].dropna()) if os.path.exists(test_path) else set()

    # Build indices
    smi_list = list(all_smiles)
    train_idx = [i for i, s in enumerate(smi_list) if s in train_smi]
    val_idx = [i for i, s in enumerate(smi_list) if s in val_smi]
    test_idx = [i for i, s in enumerate(smi_list) if s in test_smi]

    if not train_idx or (not val_idx and not test_idx):
        return None

    train_embs = all_embs[train_idx]
    train_labels = all_labels[train_idx]

    def _predict_split(split_idx):
        y_true, y_pred = [], []
        for idx in split_idx:
            emb = all_embs[idx]
            label = int(all_labels[idx])
            exclude = None
            if idx in train_idx:
                exclude = train_idx.index(idx)
            pred, _, _ = cosine_knn_predict(emb, train_embs, train_labels, k, exclude_idx=exclude)
            y_true.append(label)
            y_pred.append(pred)
        if not y_true:
            return None, None
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, zero_division=0)

    val_acc, val_f1 = _predict_split(val_idx)
    test_acc, test_f1 = _predict_split(test_idx)

    result = {
        "task": task,
        "k": k,
        "n_train": len(train_idx),
    }
    if val_acc is not None:
        result.update({"val_accuracy": round(val_acc, 4), "val_f1": round(val_f1, 4), "n_val": len(val_idx)})
    if test_acc is not None:
        result.update({"test_accuracy": round(test_acc, 4), "test_f1": round(test_f1, 4), "n_test": len(test_idx)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=27)
    args = parser.parse_args()

    tasks = []
    for d in sorted(os.listdir(CACHE_DIR)):
        if os.path.exists(os.path.join(CACHE_DIR, d, "embeddings.npz")):
            tasks.append(d)

    print(f"Found {len(tasks)} tasks with embeddings")
    results = {}
    for task in tasks:
        if task in EXCLUDE_DATASETS:
            continue
        print(f"  {task}...", end=" ", flush=True)
        metrics = evaluate_task(task, k=args.k)
        if metrics:
            results[task] = metrics
            parts = []
            if "val_accuracy" in metrics:
                parts.append(f"val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")
            if "test_accuracy" in metrics:
                parts.append(f"test_acc={metrics['test_accuracy']:.4f}, test_f1={metrics['test_f1']:.4f}")
            print(", ".join(parts))
        else:
            print("skipped")

    out_path = os.path.join(CACHE_DIR, "knn_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
