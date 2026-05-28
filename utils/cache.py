"""Shared cache helpers for therapeutic chemistry utilities."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PACKAGE_DIR / "cache"


def cache_path(*parts: str) -> Path:
    """Return a path under the therapeutic-tools cache directory."""
    return CACHE_DIR.joinpath(*parts)


@lru_cache(maxsize=None)
def load_jsonl_by_key(path: str, key: str = "smiles") -> dict[str, dict[str, Any]]:
    """Load a JSONL cache into a dict keyed by one entry field."""
    cache: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            value = entry.get(key)
            if isinstance(value, str):
                cache[value] = entry
    return cache


@lru_cache(maxsize=None)
def load_string_cache(path: str) -> dict[str, str]:
    """Load a JSONL string-output cache keyed by ``smiles``."""
    return {
        smiles: entry["result"]
        for smiles, entry in load_jsonl_by_key(path, "smiles").items()
        if isinstance(entry.get("result"), str)
    }


@lru_cache(maxsize=None)
def load_csv_by_key(path: str, key: str) -> dict[str, dict[str, str]]:
    """Load a CSV cache into a dict keyed by one column."""
    cache: dict[str, dict[str, str]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row.get(key)
            if value:
                cache[value] = row
    return cache


def lookup_jsonl(path: str | Path, smiles: str) -> Optional[dict[str, Any]]:
    """Look up a SMILES entry from a JSONL cache."""
    return load_jsonl_by_key(str(path), "smiles").get(smiles)


def lookup_string(path: str | Path, smiles: str) -> Optional[str]:
    """Look up rendered text from a JSONL string cache."""
    return load_string_cache(str(path)).get(smiles)


def get_or_compute(
    smiles: str,
    cache_lookup: Callable[[str], Optional[Any]],
    compute: Callable[[], Any],
) -> Any:
    """Return a cached value when present, otherwise compute it."""
    cached = cache_lookup(smiles)
    if cached is not None:
        return cached
    return compute()
