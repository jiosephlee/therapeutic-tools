#!/usr/bin/env python3
"""
Build GLORYx metabolite prediction cache for TDC tasks via the NERDD REST API.

Collects unique SMILES from 8 metabolism-relevant TDC tasks, submits them in
batches to GLORYx, and saves results as a JSONL cache.

Target tasks:
  CYP2C9/2D6/3A4_Substrate, DILI, AMES, Carcinogens_Lagunin, ClinTox, Bioavailability_Ma

Usage:
  python build_gloryx_cache.py [--batch-size 50] [--delay 10] [--resume]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

# NERDD API base
API_BASE = "https://nerdd.univie.ac.at/api"
GLORYX_JOBS = f"{API_BASE}/gloryx/jobs"

TARGET_TASKS = [
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin",
    "hERG",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "AMES",
]

# Paths
SCRIPT_DIR = Path(__file__).parent
TOOLS_DIR = SCRIPT_DIR.parent
REPO_ROOT = TOOLS_DIR.parent.parent.parent  # OpenRLHF-Tools
TDC_RAW = REPO_ROOT / "data" / "tdc" / "raw"
CACHE_DIR = TOOLS_DIR / "cache"
CACHE_FILE = CACHE_DIR / "gloryx_cache.jsonl"
# Existing cache from prior work (different path)
EXISTING_CACHE = Path("/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/tools/therapeutic_tools/cache/gloryx_cache.jsonl")


def collect_smiles() -> list:
    """Collect and deduplicate SMILES from target TDC tasks."""
    all_smiles = set()
    for task in TARGET_TASKS:
        task_dir = TDC_RAW / task
        for split in ["train", "valid", "test"]:
            fp = task_dir / f"{split}.csv"
            if fp.exists():
                with open(fp) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_smiles.add(row["Drug"])
    return sorted(all_smiles)


def load_existing_cache() -> set:
    """Load already-cached SMILES from both current and legacy cache files."""
    cached = set()
    for cache_path in [CACHE_FILE, EXISTING_CACHE]:
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        cached.add(entry["smiles"])
                    except Exception:
                        pass
    return cached


def copy_existing_cache_entries(needed_smiles: set):
    """Copy relevant entries from legacy cache to current cache file."""
    if not EXISTING_CACHE.exists():
        return 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    # Load current cache SMILES to avoid duplicates
    current = set()
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            for line in f:
                try:
                    current.add(json.loads(line)["smiles"])
                except Exception:
                    pass
    with open(EXISTING_CACHE) as src, open(CACHE_FILE, "a") as dst:
        for line in src:
            try:
                entry = json.loads(line)
                smi = entry["smiles"]
                if smi in needed_smiles and smi not in current:
                    dst.write(line)
                    copied += 1
            except Exception:
                pass
    return copied


def submit_batch(smiles_list: list, max_retries: int = 3) -> str:
    """Submit a batch of SMILES to GLORYx API. Returns job_id."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                GLORYX_JOBS,
                data={"inputs": smiles_list},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["id"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Submit failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def wait_for_job(job_id: str, poll_interval: int = 10, timeout: int = 600) -> bool:
    """Poll job status until completed. Returns True if successful."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(
                f"{API_BASE}/jobs/{job_id}",
                timeout=15,
            )
            response.raise_for_status()
            status = response.json()
            job_status = status.get("status", "unknown")
            processed = status.get("num_entries_processed", "?")
            total = status.get("num_entries_total", "?")
            print(f"  Job {job_id}: {job_status} ({processed}/{total})", end="\r")

            if job_status == "completed":
                print()
                return True
            elif job_status in ("failed", "error"):
                print(f"\n  Job {job_id} failed!")
                return False
        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(poll_interval)

    print(f"\n  Job {job_id} timed out after {timeout}s")
    return False


def fetch_results(job_id: str) -> list:
    """Fetch all paginated results for a completed job."""
    all_results = []
    page = 1
    while True:
        try:
            response = requests.get(
                f"{API_BASE}/jobs/{job_id}/results",
                params={"page": page},
                timeout=15,
            )
            if response.status_code == 404:
                break
            response.raise_for_status()
            data = response.json()
            results = data.get("data", [])
            if not results:
                break
            all_results.extend(results)

            # Check if there are more pages
            num_pages = data.get("job", {}).get("num_pages_total")
            if num_pages and page >= num_pages:
                break
            page += 1
        except Exception as e:
            print(f"  Fetch error on page {page}: {e}")
            break

    return all_results


def delete_job(job_id: str):
    """Clean up job after fetching results."""
    try:
        requests.delete(f"{API_BASE}/jobs/{job_id}", timeout=10)
    except Exception:
        pass


def save_results(smiles_list: list, results: list):
    """Save results to JSONL cache, matching existing format.

    Format: {"smiles": "...", "metabolites": [{"metabolite_smiles": ...,
             "reaction_type": ..., "priority_score": ..., "rank": ...}, ...]}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Group results by mol_id (input order index)
    from collections import defaultdict
    by_mol_id = defaultdict(list)
    for r in results:
        mol_id = r.get("mol_id", 0)
        by_mol_id[mol_id].append({
            "metabolite_smiles": r.get("metabolite_smiles"),
            "reaction_type": r.get("reaction_type"),
            "priority_score": r.get("priority_score"),
            "rank": r.get("rank"),
        })

    with open(CACHE_FILE, "a") as f:
        for idx, smi in enumerate(smiles_list):
            metabolites = by_mol_id.get(idx, [
                {"metabolite_smiles": None, "reaction_type": None,
                 "priority_score": None, "rank": None}
            ])
            # Sort by rank
            metabolites.sort(key=lambda m: m.get("rank") or 999)
            entry = {"smiles": smi, "metabolites": metabolites}
            f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build GLORYx cache for TDC tasks")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="SMILES per API request (default: 50)")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds between batches (default: 10)")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between status polls (default: 10)")
    parser.add_argument("--job-timeout", type=int, default=600,
                        help="Max seconds to wait per job (default: 600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just count SMILES, don't submit")
    args = parser.parse_args()

    print("Collecting SMILES from target tasks...")
    all_smiles = collect_smiles()
    print(f"Total unique SMILES: {len(all_smiles)}")

    # Copy relevant entries from legacy cache
    needed = set(all_smiles)
    copied = copy_existing_cache_entries(needed)
    if copied:
        print(f"Copied {copied} entries from existing cache at {EXISTING_CACHE}")

    # Always skip already-cached (resume is default behavior)
    cached = load_existing_cache()
    all_smiles = [s for s in all_smiles if s not in cached]
    print(f"Already cached: {len(cached)}, remaining: {len(all_smiles)}")

    if not all_smiles:
        print("Nothing to do!")
        return

    n_batches = (len(all_smiles) + args.batch_size - 1) // args.batch_size
    est_time = n_batches * (args.delay + 30)  # rough estimate
    print(f"Will submit {n_batches} batches of ~{args.batch_size}")
    print(f"Estimated minimum time: {est_time // 60}m {est_time % 60}s")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    succeeded = 0
    failed = 0

    for i in range(0, len(all_smiles), args.batch_size):
        batch = all_smiles[i : i + args.batch_size]
        batch_num = i // args.batch_size + 1
        print(f"\nBatch {batch_num}/{n_batches} ({len(batch)} SMILES)...")

        try:
            job_id = submit_batch(batch)
            print(f"  Submitted job: {job_id}")

            if wait_for_job(job_id, poll_interval=args.poll_interval,
                            timeout=args.job_timeout):
                results = fetch_results(job_id)
                print(f"  Got {len(results)} result entries")
                save_results(batch, results)
                succeeded += len(batch)
            else:
                failed += len(batch)
                print(f"  Batch {batch_num} failed — skipping")

            delete_job(job_id)

        except Exception as e:
            print(f"  Batch {batch_num} error: {e}")
            failed += len(batch)

        # Rate limit courtesy
        if i + args.batch_size < len(all_smiles):
            print(f"  Waiting {args.delay}s before next batch...")
            time.sleep(args.delay)

    print(f"\nDone! Succeeded: {succeeded}, Failed: {failed}")
    print(f"Cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()
