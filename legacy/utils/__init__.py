"""Legacy utility namespace.

Utilities remain importable here for compatibility, but public tools route
through :mod:`therapeutic_tools.api`.
"""

from __future__ import annotations

import importlib
import sys

_LEGACY_MODULES = (
    "adme",
    "cache",
    "calculator",
    "decision_tree",
    "display_names",
    "electronic",
    "endpoints",
    "functional_groups",
    "group_string_cache",
    "metadata_cache",
    "metabolism",
    "molecule_profile",
    "ring_systems",
    "safety",
    "salts",
    "scaffold",
    "similarity",
    "solubility",
    "three_d",
    "trim",
)

for _name in _LEGACY_MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"therapeutic_tools.utils.{_name}")

__all__ = list(_LEGACY_MODULES)
