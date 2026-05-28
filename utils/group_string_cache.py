"""String-output caches for grouped therapeutic-tool sections.

These caches store rendered text keyed by the exact input SMILES string.
They are intended to make modular ``get_features`` assembly cheap without
changing the visible output format.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Optional


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
_CACHE_PATHS: dict[str, str] = {
    "ring_systems": os.path.join(_CACHE_DIR, "ring_systems_cache.jsonl"),
    "v17_molecular_profile": os.path.join(_CACHE_DIR, "v17_molecular_profile_cache.jsonl"),
    "v17_ionization_and_solubility": os.path.join(_CACHE_DIR, "v17_ionization_and_solubility_cache.jsonl"),
    "v17_structure_and_topology": os.path.join(_CACHE_DIR, "v17_structure_and_topology_cache.jsonl"),
}


def cache_path(name: str) -> str:
    path = _CACHE_PATHS.get(name)
    if path is None:
        raise KeyError(f"Unknown string cache: {name}")
    return path


@lru_cache(maxsize=None)
def _load_cache(name: str) -> Dict[str, str]:
    path = cache_path(name)
    cache: Dict[str, str] = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            smiles = entry.get("smiles")
            result = entry.get("result")
            if isinstance(smiles, str) and isinstance(result, str):
                cache[smiles] = result
    return cache


def lookup(name: str, smiles: str) -> Optional[str]:
    return _load_cache(name).get(smiles)


def clear(name: str | None = None) -> None:
    _load_cache.cache_clear()
