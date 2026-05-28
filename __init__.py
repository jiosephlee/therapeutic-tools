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
    "get_molecular_properties": (f"{_BASE_PACKAGE}.tools.v10", "get_molecular_properties"),
    "get_similar_neighbors": (f"{_BASE_PACKAGE}.tools.v10", "get_similar_neighbors"),
    "get_features": (f"{_BASE_PACKAGE}.tools.v11", "get_features"),
    "get_neighbors": (f"{_BASE_PACKAGE}.tools.v11", "get_neighbors"),
    "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.tools.v15", "get_mol_properties_and_fg"),
    "compare_similar_mols": (f"{_BASE_PACKAGE}.tools.v15", "compare_similar_mols"),
}

_VERSION_FUNCTIONS = {
    "v15_no_neighbor": {
        "get_mol_properties_and_fg": (f"{_BASE_PACKAGE}.tools.v15_no_neighbor", "get_mol_properties_and_fg"),
    },
    "v15_neighbor_only": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.tools.v15_neighbor_only", "compare_similar_mols"),
    },
    "v15_neighbor_only_4": {
        "compare_similar_mols": (f"{_BASE_PACKAGE}.tools.v15_neighbor_only_4", "compare_similar_mols"),
    },
    "v17_get_features_only": {
        "get_features": (f"{_BASE_PACKAGE}.tools.v17_get_features_only", "get_features"),
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


__all__ = ["get_function_by_name"]
