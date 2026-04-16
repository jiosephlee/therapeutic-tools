"""Version 14 therapeutic tools without neighbor lookup (``get_features`` only).

The feature vocabulary and ``get_features`` implementation are identical to
:mod:`openrlhf.tools.therapeutic_tools.v14`.  Nearest-neighbor tools
(``get_neighbors``, ``get_neighbors_*``) are omitted from this surface for
datasets and registries that must not expose similarity-based retrieval.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .v14 import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    V14_ADDITIONAL_FEATURES,
    get_features,
    GET_FEATURES_TOOL as _V14_GET_FEATURES_TOOL,
)

GET_FEATURES_TOOL: Dict[str, Any] = copy.deepcopy(_V14_GET_FEATURES_TOOL)
GET_FEATURES_TOOL["function"]["description"] = (
    "Return requested molecular features for a SMILES string (v14 no-neighbor: same "
    "vocabulary as v14 — v12 base minus metabolites plus neutral fraction, Labute ASA, "
    "NOCount, and carbocycle counts; neighbor / similarity tools are not available)."
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "V14_ADDITIONAL_FEATURES",
    "get_features",
    "GET_FEATURES_TOOL",
]
