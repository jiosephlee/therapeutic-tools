"""
Build coarse ChEMBL top-K Morgan candidates for official_v15 molecules.

Uses ChEMBL's official FPSim2 HDF5 database for a fast Morgan-only pass, then
writes one JSON line per query molecule plus a union-of-candidates file.

Outputs under therapeutic_tools/cache/chembl/official_v15/:
  - coarse_topk_k{K}.jsonl
  - coarse_topk_k{K}_union_molregnos.txt
  - coarse_topk_k{K}_stats.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from FPSim2 import FPSim2Engine

from chembl_cache_utils import (
    CHEMBL_H5_PATH,
    CHEMBL_OFFICIAL_V15_DIR,
    ensure_directory,
    load_completed_queries,
    load_official_v15_queries,
)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = build_args()
    if not CHEMBL_H5_PATH.exists():
        raise FileNotFoundError(f"Missing ChEMBL H5 database: {CHEMBL_H5_PATH}")

    ensure_directory(CHEMBL_OFFICIAL_V15_DIR)
    out_jsonl = CHEMBL_OFFICIAL_V15_DIR / f"coarse_topk_k{args.top_k}.jsonl"
    out_union = CHEMBL_OFFICIAL_V15_DIR / f"coarse_topk_k{args.top_k}_union_molregnos.txt"
    out_stats = CHEMBL_OFFICIAL_V15_DIR / f"coarse_topk_k{args.top_k}_stats.json"

    if args.overwrite:
        for path in (out_jsonl, out_union, out_stats):
            if path.exists():
                path.unlink()

    completed = load_completed_queries(out_jsonl)
    queries = load_official_v15_queries()
    if args.limit is not None:
        queries = queries[: args.limit]

    remaining = [row for row in queries if row["query_smiles"] not in completed]
    print(f"Loaded {len(queries)} official_v15 queries; {len(remaining)} remaining")
    engine = FPSim2Engine(str(CHEMBL_H5_PATH), in_memory_fps=False)

    start = time.time()
    processed = 0
    union_molregnos: set[int] = set()
    if out_union.exists():
        with out_union.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    union_molregnos.add(int(line))

    with out_jsonl.open("a") as f:
        for row in remaining:
            query_smiles = str(row["query_smiles"])
            hits = engine.on_disk_top_k(
                query_smiles,
                k=args.top_k,
                threshold=args.threshold,
                n_workers=args.n_workers,
                chunk_size=args.chunk_size or None,
            )
            neighbors = []
            for hit in hits:
                molregno = int(hit["mol_id"])
                coeff = float(hit["coeff"])
                neighbors.append(
                    {
                        "molregno": molregno,
                        "coarse_morgan_tanimoto": coeff,
                    }
                )
                union_molregnos.add(molregno)

            payload = {
                "query_smiles": query_smiles,
                "canonical_smiles": row["canonical_smiles"],
                "tasks": row["tasks"],
                "neighbors": neighbors,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"processed {processed}/{len(remaining)} "
                    f"({rate:.2f} queries/s, union={len(union_molregnos)})"
                )

    with out_union.open("w") as f:
        for molregno in sorted(union_molregnos):
            f.write(f"{molregno}\n")

    stats = {
        "top_k": args.top_k,
        "threshold": args.threshold,
        "n_queries": len(queries),
        "n_completed": len(queries) - len(remaining) + processed,
        "n_union_molregnos": len(union_molregnos),
        "h5_path": str(CHEMBL_H5_PATH),
    }
    out_stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
