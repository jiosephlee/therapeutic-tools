"""
Summarize the final ChEMBL top-neighbor cache for official_v15.

Outputs:
  - aggregate JSON summary
  - per-task CSV with one row per query-neighbor pair
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from chembl_cache_utils import CHEMBL_OFFICIAL_V15_DIR, ensure_directory


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--final-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    in_path = CHEMBL_OFFICIAL_V15_DIR / f"top{args.final_k}_weighted_neighbors_k{args.top_k}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing final neighbor cache: {in_path}")

    out_dir = ensure_directory(CHEMBL_OFFICIAL_V15_DIR / "reports")
    summary_path = out_dir / f"summary_top{args.final_k}_k{args.top_k}.json"
    rows_path = out_dir / f"per_neighbor_top{args.final_k}_k{args.top_k}.csv"
    task_summary_path = out_dir / f"per_task_top{args.final_k}_k{args.top_k}.csv"

    rows: list[dict] = []
    task_counter: Counter[str] = Counter()
    top1_counter: Counter[str] = Counter()
    weighted_by_task: dict[str, list[float]] = defaultdict(list)

    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query_smiles = payload["query_smiles"]
            tasks = payload.get("tasks", [])
            neighbors = payload.get("neighbors", [])
            for task in tasks:
                task_counter[task] += 1
            if neighbors:
                top1 = neighbors[0].get("chembl_id")
                if isinstance(top1, str):
                    top1_counter[top1] += 1
            for neighbor in neighbors:
                row = {
                    "query_smiles": query_smiles,
                    "tasks": "|".join(tasks),
                    **neighbor,
                }
                rows.append(row)
                for task in tasks:
                    score = neighbor.get("weighted_tanimoto")
                    if score is not None:
                        weighted_by_task[task].append(float(score))

    df = pd.DataFrame(rows)
    df.to_csv(rows_path, index=False)

    per_task_rows = []
    for task, n_queries in sorted(task_counter.items()):
        scores = weighted_by_task.get(task, [])
        per_task_rows.append(
            {
                "task": task,
                "n_queries": n_queries,
                "n_neighbor_rows": len(scores),
                "mean_weighted_tanimoto": (sum(scores) / len(scores)) if scores else None,
                "median_weighted_tanimoto": pd.Series(scores).median() if scores else None,
                "max_weighted_tanimoto": max(scores) if scores else None,
            }
        )
    pd.DataFrame(per_task_rows).to_csv(task_summary_path, index=False)

    summary = {
        "input_path": str(in_path),
        "n_queries": int(df["query_smiles"].nunique()) if not df.empty else 0,
        "n_neighbor_rows": int(len(df)),
        "n_unique_neighbor_chembl_ids": int(df["chembl_id"].nunique()) if not df.empty else 0,
        "task_query_counts": dict(sorted(task_counter.items())),
        "top1_most_common": top1_counter.most_common(25),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {task_summary_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
