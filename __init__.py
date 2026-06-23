"""Therapeutic chemistry tools.

Package layout:
- :mod:`tools.therapeutic_tools.utils` contains canonical endpoints and
  semantic chemistry utilities.
- :mod:`tools.therapeutic_tools.tools` contains versioned LLM-facing surfaces.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable

_BASE_PACKAGE = __name__

_DEFAULT_FUNCTIONS = {
    "get_primitive": (f"{_BASE_PACKAGE}.api", "get_primitive"),
    "get_primitives": (f"{_BASE_PACKAGE}.api", "get_primitives"),
    "get_bundle": (f"{_BASE_PACKAGE}.api", "get_bundle"),
    "get_all_features": (f"{_BASE_PACKAGE}.api", "get_all_features"),
    "get_molecule_profile": (f"{_BASE_PACKAGE}.utils.molecule_profile", "get_molecule_profile"),
    "analyze_functional_groups": (f"{_BASE_PACKAGE}.utils.functional_groups", "analyze_functional_groups"),
    "analyze_ring_systems": (f"{_BASE_PACKAGE}.utils.ring_systems", "analyze_ring_systems"),
    "assess_adme_properties": (f"{_BASE_PACKAGE}.utils.adme", "assess_adme_properties"),
    "get_3d_properties": (f"{_BASE_PACKAGE}.utils.three_d", "get_3d_properties"),
    "screen_structural_alerts": (f"{_BASE_PACKAGE}.utils.safety", "screen_structural_alerts"),
    "find_similar_molecules": (f"{_BASE_PACKAGE}.utils.similarity", "find_similar_molecules"),
    "remove_salts": (f"{_BASE_PACKAGE}.utils.salts", "remove_salts"),
    "evaluate_arithmetic": (f"{_BASE_PACKAGE}.utils.calculator", "evaluate_arithmetic"),
    "get_electronic_properties": (f"{_BASE_PACKAGE}.utils.electronic", "get_electronic_properties"),
    "predict_metabolites": (f"{_BASE_PACKAGE}.utils.metabolism", "predict_metabolites"),
    "get_scaffold": (f"{_BASE_PACKAGE}.utils.scaffold", "get_scaffold"),
    "predict_solubility": (f"{_BASE_PACKAGE}.utils.solubility", "predict_solubility"),
    "decision_tree_analysis": (f"{_BASE_PACKAGE}.utils.decision_tree", "decision_tree_analysis"),
    "get_features": (f"{_BASE_PACKAGE}.tools.v17", "get_features"),
    "get_neighbors": (f"{_BASE_PACKAGE}.tools.v17", "get_neighbors"),
}

_VERSION_FUNCTIONS = {
    "v10": {
        "get_molecular_properties": (f"{_BASE_PACKAGE}.legacy.tools.v10", "get_molecular_properties"),
        "get_similar_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v10", "get_similar_neighbors"),
    },
    "legacy_v10": {
        "get_molecular_properties": (f"{_BASE_PACKAGE}.legacy.tools.v10", "get_molecular_properties"),
        "get_similar_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v10", "get_similar_neighbors"),
    },
    "v11": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_neighbors"),
    },
    "legacy_v11": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_neighbors"),
    },
    "v11_legacy": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v11", "get_neighbors"),
    },
    "v12": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v12", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v12", "get_neighbors"),
    },
    "legacy_v12": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v12", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v12", "get_neighbors"),
    },
    "v13": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v13", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v13", "get_neighbors"),
    },
    "legacy_v13": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v13", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v13", "get_neighbors"),
    },
    "v14": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v14", "get_neighbors"),
    },
    "legacy_v14": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v14", "get_neighbors"),
    },
    "v14_consolidated": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated", "get_neighbors"),
    },
    "legacy_v14_consolidated": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated", "get_neighbors"),
    },
    "v14_consolidated_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated_no_neighbor", "get_features"),
    },
    "legacy_v14_consolidated_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_consolidated_no_neighbor", "get_features"),
    },
    "v14_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_no_neighbor", "get_features"),
    },
    "legacy_v14_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v14_no_neighbor", "get_features"),
    },
    "v15": {
        "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.legacy.tools.v15", "get_mol_properties_and_fg"),
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15", "compare_similar_mols"),
    },
    "legacy_v15": {
        "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.legacy.tools.v15", "get_mol_properties_and_fg"),
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15", "compare_similar_mols"),
    },
    "v15_no_neighbor": {
        "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.legacy.tools.v15_no_neighbor", "get_mol_properties_and_fg"),
    },
    "legacy_v15_no_neighbor": {
        "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.legacy.tools.v15_no_neighbor", "get_mol_properties_and_fg"),
    },
    "v15_neighbor_only": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15_neighbor_only", "compare_similar_mols"),
    },
    "legacy_v15_neighbor_only": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15_neighbor_only", "compare_similar_mols"),
    },
    "v15_neighbor_only_4": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15_neighbor_only_4", "compare_similar_mols"),
    },
    "legacy_v15_neighbor_only_4": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.legacy.tools.v15_neighbor_only_4", "compare_similar_mols"),
    },
    "v16": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v16", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v16", "get_neighbors"),
    },
    "legacy_v16": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v16", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v16", "get_neighbors"),
    },
    "v16_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v16_no_neighbor", "get_features"),
    },
    "legacy_v16_no_neighbor": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v16_no_neighbor", "get_features"),
    },
    "v17_get_features_only": {
        "get_features": (f"{_BASE_PACKAGE}.tools.v17_get_features_only", "get_features"),
    },
    "legacy_v17_get_features_only": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v17_get_features_only", "get_features"),
    },
    "v17_get_features_only_legacy": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v17_get_features_only", "get_features"),
    },
    "legacy_v17": {
        "get_features": (f"{_BASE_PACKAGE}.legacy.tools.v17", "get_features"),
        "get_neighbors": (f"{_BASE_PACKAGE}.legacy.tools.v17", "get_neighbors"),
    },
}


def _load_function(target: tuple[str, str]) -> Callable:
    module_name, attr = target
    return getattr(importlib.import_module(module_name), attr)


def get_function_by_name(name: str):
    """Look up a tool function by name for the active tool version."""
    tool_version = os.environ.get("OPENRLHF_TOOL_VERSION")
    version_map = _VERSION_FUNCTIONS.get(tool_version, {})
    target = version_map.get(name) or _DEFAULT_FUNCTIONS.get(name)
    return _load_function(target) if target else None


def get_primitive(*args, **kwargs):
    from .api import get_primitive as _get_primitive

    return _get_primitive(*args, **kwargs)


def get_primitives(*args, **kwargs):
    from .api import get_primitives as _get_primitives

    return _get_primitives(*args, **kwargs)


def get_bundle(*args, **kwargs):
    from .api import get_bundle as _get_bundle

    return _get_bundle(*args, **kwargs)


def get_all_features(*args, **kwargs):
    from .api import get_all_features as _get_all_features

    return _get_all_features(*args, **kwargs)

__all__ = [
    "get_all_features",
    "get_bundle",
    "get_function_by_name",
    "get_primitive",
    "get_primitives",
]
