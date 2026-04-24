"""
v17-specific per-molecule cache for the analogue-quality neighbor tool.

Each JSONL line stores the precomputed artifacts the ``get_neighbors`` block
consults, keyed by canonical SMILES:

  - ``murcko_scaffold``: Bemis-Murcko scaffold SMILES
  - ``functional_groups``: sorted list of FG names (from
    ``functional_groups.analyze_functional_groups(..., simple=True)``)
  - ``size_shape``: dict with molecular_weight, heavy_atoms, ring_total,
    fraction_csp3, rotatable_bonds
  - ``mmp_single``: dict mapping context-SMILES → R-group-SMILES for all
    single-cut Hussain-Rea fragmentations
  - ``mmp_double``: dict mapping context-SMILES (with [*:1]/[*:2]) →
    variable-fragment SMILES for all double-cut fragmentations, with both
    dummy-label orderings stored

The cache is loaded once and held in memory — missing entries fall back to
runtime computation so v17 stays functional without a pre-built cache.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "v17_neighbor_cache.jsonl")


@lru_cache(maxsize=1)
def _load_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(CACHE_PATH):
        return cache
    with open(CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                smi = entry.get("smiles")
                if smi:
                    cache[smi] = entry
            except Exception:
                continue
    return cache


def lookup(smi: str) -> Optional[Dict[str, Any]]:
    """Return cached artifacts for ``smi`` (canonicalized), or None."""
    cache = _load_cache()
    if not cache:
        return None
    from .similarity import _canonicalize_smiles

    # Try raw first (exact match from training set) then canonical form.
    entry = cache.get(smi)
    if entry is not None:
        return entry
    canon = _canonicalize_smiles(smi)
    if canon is None:
        return None
    return cache.get(canon)


def compute_artifacts(smi: str) -> Dict[str, Any]:
    """Compute all v17 artifacts for a single SMILES (used by the cache builder
    and as a runtime fallback).
    """
    from .similarity import _canonicalize_smiles, _get_murcko_scaffold
    from .v17 import (
        _compute_size_shape_summary_uncached,
        _functional_group_set_uncached,
        compute_fragmentations,
    )

    canon = _canonicalize_smiles(smi) or smi
    try:
        scaffold = _get_murcko_scaffold(canon) or ""
    except Exception:
        scaffold = ""
    try:
        fgs: List[str] = sorted(_functional_group_set_uncached(canon))
    except Exception:
        fgs = []
    try:
        size_shape = _compute_size_shape_summary_uncached(canon)
    except Exception:
        size_shape = {}
    try:
        single, double = compute_fragmentations(canon)
    except Exception:
        single, double = {}, {}
    return {
        "smiles": canon,
        "murcko_scaffold": scaffold,
        "functional_groups": fgs,
        "size_shape": size_shape,
        "mmp_single": single,
        "mmp_double": double,
    }
