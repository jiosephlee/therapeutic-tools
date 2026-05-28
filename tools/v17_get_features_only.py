"""Version 17 therapeutic tools with only ``get_features`` exposed.

The feature vocabulary and grouped-feature output are identical to
``openrlhf.tools.therapeutic_tools.tools.v17``. This surface omits neighbor lookup
tools so traces can simulate a single ``get_features`` call cleanly.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .v17 import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    get_features,
    GET_FEATURES_TOOL as _V17_GET_FEATURES_TOOL,
)

GET_FEATURES_TOOL: Dict[str, Any] = copy.deepcopy(_V17_GET_FEATURES_TOOL)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "get_features",
    "GET_FEATURES_TOOL",
]
