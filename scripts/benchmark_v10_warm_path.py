#!/usr/bin/env python
"""
Benchmark warm-path latency for v10 therapeutic tools without modifying production code.

Usage:
  python openrlhf/tools/therapeutic_tools/scripts/benchmark_v10_warm_path.py

Notes:
  - Run this inside a working environment for therapeutic_tools (RDKit, etc.).
  - Sets MPLCONFIGDIR to /tmp if unset to avoid matplotlib temp-cache overhead.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from statistics import mean


if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-openrlhf"


from openrlhf.tools.therapeutic_tools.utils import metadata_cache
from openrlhf.tools.therapeutic_tools.utils.adme import assess_adme_properties
from openrlhf.tools.therapeutic_tools.utils.functional_groups import analyze_functional_groups
from openrlhf.tools.therapeutic_tools.utils.metabolism import predict_metabolites
from openrlhf.tools.therapeutic_tools.utils.molecule_profile import get_molecule_profile
from openrlhf.tools.therapeutic_tools.utils.ring_systems import analyze_ring_systems
from openrlhf.tools.therapeutic_tools.utils.safety import screen_safety
from openrlhf.tools.therapeutic_tools.utils.three_d import get_3d_properties
from openrlhf.tools.therapeutic_tools.tools.v10 import get_molecular_properties


def load_default_smiles(limit: int = 10) -> list[str]:
    path = Path(metadata_cache._METADATA_PATH)
    smiles: list[str] = []
    seen: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("Drug", "")
            if not smi or smi in seen:
                continue
            seen.add(smi)
            smiles.append(smi)
            if len(smiles) >= limit:
                break
    return smiles


def time_calls(label: str, fn, smiles_list: list[str]) -> list[float]:
    times: list[float] = []
    for idx, smiles in enumerate(smiles_list, 1):
        start = time.perf_counter()
        fn(smiles)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"{label} {idx} {elapsed:.3f}")
    print(f"{label}_total {sum(times):.3f}")
    print(f"{label}_mean {mean(times):.3f}")
    return times


def inspect_adme_cache(smiles_list: list[str]) -> None:
    print("adme_cache_status")
    keys = [
        "most_acidic_pka",
        "most_basic_pka",
        "num_acidic_sites",
        "num_basic_sites",
        "acid_sites_json",
        "base_sites_json",
        "logD_74",
    ]
    for idx, smiles in enumerate(smiles_list, 1):
        row = metadata_cache.lookup_row(smiles)
        present = [key for key in keys if row and key in row]
        print(f"{idx} {len(present)}/{len(keys)} {' '.join(present)}")


def benchmark_subtools(smiles_list: list[str]) -> None:
    tools = [
        (
            "molecule_profile",
            lambda s: get_molecule_profile(
                s,
                include_lipinski_violations=False,
                include_electronic_summary=False,
                include_quantum_properties=False,
            ),
        ),
        ("functional_groups", lambda s: analyze_functional_groups(s, simple=True)),
        ("ring_systems", analyze_ring_systems),
        ("adme", lambda s: assess_adme_properties(s, ph=7.4, simple_pka=True)),
        ("three_d", lambda s: get_3d_properties(s, include_epsa=False)),
        ("safety", screen_safety),
        ("metabolism", lambda s: predict_metabolites(s, max_metabolites=1)),
    ]

    # Warm singleton/process state on the first molecule before measuring the rest.
    for _, fn in tools:
        fn(smiles_list[0])

    for name, fn in tools:
        times: list[float] = []
        for smiles in smiles_list:
            start = time.perf_counter()
            fn(smiles)
            times.append(time.perf_counter() - start)
        print(f"{name}_total {sum(times):.3f}")
        print(f"{name}_mean {mean(times):.3f}")
        print(f"{name}_max {max(times):.3f}")
        print(f"{name}_times {' '.join(f'{x:.3f}' for x in times)}")


def main() -> None:
    smiles_list = load_default_smiles(limit=10)
    print(f"molecule_count {len(smiles_list)}")
    print("v10_serial")
    time_calls("cold", get_molecular_properties, smiles_list)
    time_calls("warm", get_molecular_properties, smiles_list)
    inspect_adme_cache(smiles_list)
    benchmark_subtools(smiles_list)


if __name__ == "__main__":
    main()
