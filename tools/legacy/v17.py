"""Legacy v17 surface that bypasses the DuckDB feature-text cache."""

from __future__ import annotations

from typing import List

from .. import v17 as _v17
from ...legacy.tools import v16 as _v16

FEATURE_NAMES = _v17.FEATURE_NAMES
FEATURE_GROUP_DESCRIPTIONS = _v17.FEATURE_GROUP_DESCRIPTIONS
FEATURE_REGISTRY = _v16.FEATURE_REGISTRY
GET_FEATURES_TOOL = _v17.GET_FEATURES_TOOL
K_NEIGHBORS = _v17.K_NEIGHBORS


def get_features(smiles: str, feature_names: List[str]) -> str:
    if not feature_names:
        return f"Error: feature_names is empty. Available: {', '.join(FEATURE_NAMES)}"
    resolved, errors = _v16._resolve_feature_names(feature_names)
    if errors:
        return (
            f"Error: Unknown feature(s): {', '.join(errors)}. "
            f"Available: {', '.join(FEATURE_NAMES)}"
        )
    return _v16._compute_features_for_smiles(smiles, resolved)


def get_neighbors(*args, **kwargs):
    return _v17.get_neighbors(*args, **kwargs)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUP_DESCRIPTIONS",
    "FEATURE_REGISTRY",
    "GET_FEATURES_TOOL",
    "K_NEIGHBORS",
    "get_features",
    "get_neighbors",
]
