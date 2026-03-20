"""
Precompute KNN accuracy and F1 per TDC task using cached embeddings.

For each task, runs leave-one-out KNN (k=27) on the training embeddings,
then evaluates on the val+test splits. Saves per-task metrics to knn_metrics.json.

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
DATA_DIR = os.path.join(CACHE_DIR, "..", "..", "..", "..", "data", "tdc", "raw")
EXCLUDE_DATASETS = {"Tox21", "HIV", "herg_central"}


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

    # Load train/val/test splits to identify which are test
    train_path = os.path.join(os.path.abspath(DATA_DIR), task, "train.csv")
    val_path = os.path.join(os.path.abspath(DATA_DIR), task, "val.csv")
    test_path = os.path.join(os.path.abspath(DATA_DIR), task, "test.csv")

    if not os.path.exists(train_path):
        return None

    train_smi = set(pd.read_csv(train_path)["Drug"].dropna())
    val_smi = set(pd.read_csv(val_path)["Drug"].dropna()) if os.path.exists(val_path) else set()
    test_smi = set(pd.read_csv(test_path)["Drug"].dropna()) if os.path.exists(test_path) else set()

    # Build indices
    smi_list = list(all_smiles)
    train_idx = [i for i, s in enumerate(smi_list) if s in train_smi]
    eval_idx = [i for i, s in enumerate(smi_list) if s in (val_smi | test_smi)]

    if not train_idx or not eval_idx:
        # Fallback: leave-one-out on all data
        train_idx = list(range(len(smi_list)))
        eval_idx = train_idx

    train_embs = all_embs[train_idx]
    train_labels = all_labels[train_idx]

    y_true = []
    y_pred = []
    for idx in eval_idx:
        emb = all_embs[idx]
        label = int(all_labels[idx])

        # If this sample is also in train, exclude it
        exclude = None
        if idx in train_idx:
            exclude = train_idx.index(idx)

        pred, _, _ = cosine_knn_predict(emb, train_embs, train_labels, k, exclude_idx=exclude)
        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "task": task,
        "k": k,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
    }


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
            print(f"acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        else:
            print("skipped")

    out_path = os.path.join(CACHE_DIR, "knn_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
