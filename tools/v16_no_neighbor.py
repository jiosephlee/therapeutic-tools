"""Version 16 therapeutic tools without neighbor lookup.

The feature vocabulary and ``get_features`` implementation are identical to
:mod:`openrlhf.tools.therapeutic_tools.tools.v16`. Nearest-neighbor tools
(``get_neighbors``, ``get_neighbors_*``) are omitted from this surface.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .v16 import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    get_features,
    GET_FEATURES_TOOL as _V16_GET_FEATURES_TOOL,
)

GET_FEATURES_TOOL: Dict[str, Any] = copy.deepcopy(_V16_GET_FEATURES_TOOL)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "get_features",
    "GET_FEATURES_TOOL",
]
