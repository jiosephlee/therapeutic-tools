"""Legacy LLM-facing therapeutic tool surfaces.

Only the current v17 surfaces are DuckDB-first. Older versioned tools have been
archived under ``therapeutic_tools.legacy.tools``. This package keeps legacy
``therapeutic_tools.tools.legacy`` import paths available.
"""

from __future__ import annotations

import importlib
import sys

_LEGACY_MODULES = {
    "v10": "therapeutic_tools.legacy.tools.v10",
    "v11": "therapeutic_tools.legacy.tools.v11",
    "v12": "therapeutic_tools.legacy.tools.v12",
    "v13": "therapeutic_tools.legacy.tools.v13",
    "v14": "therapeutic_tools.legacy.tools.v14",
    "v14_consolidated": "therapeutic_tools.legacy.tools.v14_consolidated",
    "v14_consolidated_no_neighbor": "therapeutic_tools.legacy.tools.v14_consolidated_no_neighbor",
    "v14_no_neighbor": "therapeutic_tools.legacy.tools.v14_no_neighbor",
    "v15": "therapeutic_tools.legacy.tools.v15",
    "v15_neighbor_only": "therapeutic_tools.legacy.tools.v15_neighbor_only",
    "v15_neighbor_only_4": "therapeutic_tools.legacy.tools.v15_neighbor_only_4",
    "v15_no_neighbor": "therapeutic_tools.legacy.tools.v15_no_neighbor",
    "v16": "therapeutic_tools.legacy.tools.v16",
    "v16_no_neighbor": "therapeutic_tools.legacy.tools.v16_no_neighbor",
    "v17": "therapeutic_tools.tools.legacy.v17",
    "v17_get_features_only": "therapeutic_tools.tools.legacy.v17_get_features_only",
}

for _name, _target in _LEGACY_MODULES.items():
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(_target)

__all__ = sorted(_LEGACY_MODULES)
