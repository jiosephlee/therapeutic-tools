"""Version 17 therapeutic tools."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..primitive_registry import (
    FEATURE_GROUP_DESCRIPTIONS,
    PUBLIC_FEATURE_NAMES as FEATURE_NAMES,
    feature_names_description_text,
    resolve_feature_names,
)
from ._v17_neighbors import K_NEIGHBORS, get_neighbors

FEATURE_REGISTRY: Dict[str, Any] = {}


def get_features(smiles: str, feature_names: List[str]) -> str:
    if not feature_names:
        return f"Error: feature_names is empty. Available: {', '.join(FEATURE_NAMES)}"
    resolved, errors = resolve_feature_names(feature_names)
    if errors:
        return (
            f"Error: Unknown feature(s): {', '.join(errors)}. "
            f"Available: {', '.join(FEATURE_NAMES)}"
        )
    try:
        from ..api import get_feature_text

        return get_feature_text(smiles, resolved)
    except Exception as exc:
        if os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND", "").lower() == "duckdb_only":
            raise
        return f"get_features: Error - {exc}"


GET_FEATURES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_features",
        "description": (
            "Compute chemistry evidence for a molecule across selected feature "
            "groups: 'molecular_profile' (size, shape, logP, TPSA, H-bond "
            "donors/acceptors, electronic descriptors), 'ionization_and_solubility' "
            "(pKa, dominant form at pH 7.4, logD, aqueous solubility), "
            "'structure_and_topology' (functional groups), 'alert_screening' "
            "(structural-alert categories for toxicity/reactivity). Request only "
            "the groups needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Molecule to analyze, provided as a SMILES string.",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": feature_names_description_text(),
                },
            },
            "required": ["smiles", "feature_names"],
            "additionalProperties": False,
        },
    },
}


GET_NEIGHBORS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_neighbors",
        "description": (
            "Find the 3 most structurally similar labeled molecules to the query. "
            "For each neighbor returns: Tanimoto similarity, label, Bemis-Murcko "
            "scaffold match, MCS size, the substituent/linker change from query to "
            "neighbor (MMP transformation), shared vs query-only vs neighbor-only "
            "functional groups, and size/shape deltas (MW, heavy atoms, rings, Fsp3, "
            "rotatable bonds). Use to judge whether a neighbor's label transfers to "
            "the query. For logP, TPSA, pKa, solubility, or alerts, use get_features."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Query molecule, provided as a SMILES string.",
                },
                "task_name": {
                    "type": "string",
                    "description": "TDC task name to use for neighbor retrieval.",
                },
            },
            "required": ["smiles", "task_name"],
            "additionalProperties": False,
        },
    },
}

TASK_NEIGHBOR_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}
TASK_NEIGHBOR_CALLABLES: Dict[str, Any] = {}

__all__ = [
    "FEATURE_GROUP_DESCRIPTIONS",
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "GET_FEATURES_TOOL",
    "GET_NEIGHBORS_TOOL",
    "K_NEIGHBORS",
    "TASK_NEIGHBOR_CALLABLES",
    "TASK_NEIGHBOR_TOOL_SCHEMAS",
    "get_features",
    "get_neighbors",
]
