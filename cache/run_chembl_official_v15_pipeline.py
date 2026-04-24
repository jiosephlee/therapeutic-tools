"""
Driver for the ChEMBL -> official_v15 neighbor pipeline.

Runs:
  1. coarse candidate generation
  2. reduced-subset metadata/fingerprint build
  3. final weighted rerank
  4. summary report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_step(script: str, *extra_args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *extra_args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = build_args()
    common = [f"--top-k={args.top_k}"]
    if args.limit is not None:
        common.append(f"--limit={args.limit}")
    if args.overwrite:
        common.append("--overwrite")

    run_step(
        "build_chembl_coarse_candidates.py",
        *common,
        f"--n-workers={args.n_workers}",
    )
    run_step(
        "build_chembl_subset_fingerprints.py",
        f"--top-k={args.top_k}",
        f"--radius={args.radius}",
        f"--fp-size={args.fp_size}",
        *("--overwrite",) if args.overwrite else (),
    )
    rerank_args = [
        f"--top-k={args.top_k}",
        f"--final-k={args.final_k}",
        f"--radius={args.radius}",
        f"--fp-size={args.fp_size}",
    ]
    if args.limit is not None:
        rerank_args.append(f"--limit={args.limit}")
    if args.overwrite:
        rerank_args.append("--overwrite")
    run_step("build_chembl_official_v15_top10.py", *rerank_args)
    run_step(
        "build_chembl_official_v15_report.py",
        f"--top-k={args.top_k}",
        f"--final-k={args.final_k}",
    )


if __name__ == "__main__":
    main()
