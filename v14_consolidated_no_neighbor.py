"""Version 14 consolidated therapeutic tools without neighbor lookup.

The feature vocabulary and ``get_features`` implementation are identical to
:mod:`openrlhf.tools.therapeutic_tools.v14_consolidated`. Nearest-neighbor tools
(``get_neighbors``, ``get_neighbors_*``) are omitted from this surface for
datasets and registries that must not expose similarity-based retrieval.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .v14_consolidated import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    get_features,
    GET_FEATURES_TOOL as _V14_CONSOLIDATED_GET_FEATURES_TOOL,
)

GET_FEATURES_TOOL: Dict[str, Any] = copy.deepcopy(_V14_CONSOLIDATED_GET_FEATURES_TOOL)
GET_FEATURES_TOOL["function"]["description"] = (
    "Analyze a molecule and return selected groups of medicinal-chemistry evidence, "
    "such as physicochemical properties; pKa, ionization, logD, and solubility; "
    "functional groups and ring systems; electronic and structural-alert liabilities; "
    "or 3D shape and flexibility features."
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "get_features",
    "GET_FEATURES_TOOL",
]
