"""
Shared metadata cache — loads tdc_metadata_consolidated.csv once.

All tools should call `lookup(smiles, prop)` before computing a descriptor
from scratch. The CSV contains ~48 precomputed columns for ~45k molecules.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict
from functools import lru_cache

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_METADATA_PATH = os.path.join(_CACHE_DIR, "tdc_metadata_consolidated.csv")


@lru_cache(maxsize=1)
def _load_metadata() -> Optional[pd.DataFrame]:
    """Load the consolidated metadata CSV indexed by Drug (SMILES)."""
    if not os.path.exists(_METADATA_PATH):
        return None
    df = pd.read_csv(_METADATA_PATH)
    if "Drug" not in df.columns:
        return None
    df = df.set_index("Drug")
    return df


def lookup(smiles: str, prop: str) -> Optional[float]:
    """Look up a single numerical property from the cached metadata.

    Returns the value if found and non-NaN, otherwise None.
    """
    meta = _load_metadata()
    if meta is None or smiles not in meta.index or prop not in meta.columns:
        return None
    val = meta.at[smiles, prop]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val) if pd.notna(val) else None


def lookup_row(smiles: str) -> Optional[Dict[str, float]]:
    """Look up all cached properties for a SMILES. Returns dict or None."""
    meta = _load_metadata()
    if meta is None or smiles not in meta.index:
        return None
    row = meta.loc[smiles]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {k: float(v) for k, v in row.items() if pd.notna(v)}
