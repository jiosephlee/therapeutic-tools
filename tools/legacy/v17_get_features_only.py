"""Legacy v17 get_features-only surface that bypasses DuckDB."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .v17 import FEATURE_NAMES, FEATURE_REGISTRY, get_features
from ..v17 import GET_FEATURES_TOOL as _V17_GET_FEATURES_TOOL

GET_FEATURES_TOOL: Dict[str, Any] = copy.deepcopy(_V17_GET_FEATURES_TOOL)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "get_features",
    "GET_FEATURES_TOOL",
]

