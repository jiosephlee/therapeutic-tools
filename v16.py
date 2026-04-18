"""
Version 16 therapeutic tools.

This variant further consolidates :mod:`v14_consolidated` into four grouped
feature buckets while preserving the same underlying v14 evidence surface:

  - ``molecular_profile``: physicochemical + complexity + electronic properties + PMI-based 3D shape
  - ``ionization_and_solubility``: pKa + ionization + logD + solubility
  - ``structure_and_topology``: functional groups + ring systems
  - ``alert_screening``: structural alerts
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .similarity import TASKS, TASK_ALIASES
from .v11 import _fuzzy_match, _label_str, _resolve_task_name
from .v14_consolidated import (
    _compute_physicochemical,
    _compute_complexity,
    _compute_electronic,
    _compute_ionization,
    _compute_logd,
    _compute_pka,
    _compute_solubility,
)


FEATURE_NAMES: List[str] = [
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
]

_FEATURE_ALIASES: Dict[str, List[str]] = {
    "molecularprofile": ["molecular_profile"],
    "profile": ["molecular_profile"],
    "molecularproperties": ["molecular_profile"],
    "core": ["molecular_profile"],
    "coreproperties": ["molecular_profile"],
    "physicochemical": ["molecular_profile"],
    "complexity": ["molecular_profile"],
    "electronic": ["molecular_profile"],
    "shape": ["molecular_profile"],
    "flexibility": ["molecular_profile"],
    "3d": ["molecular_profile"],
    "3dproperties": ["molecular_profile"],
    "shapeandflexibility": ["molecular_profile"],
    "developability": ["molecular_profile", "ionization_and_solubility"],
    "ionization": ["ionization_and_solubility"],
    "pka": ["ionization_and_solubility"],
    "logd": ["ionization_and_solubility"],
    "solubility": ["ionization_and_solubility"],
    "ionizationandsolubility": ["ionization_and_solubility"],
    "structure": ["structure_and_topology"],
    "topology": ["structure_and_topology"],
    "functionalgroups": ["structure_and_topology"],
    "ringsystems": ["structure_and_topology"],
    "structureandtopology": ["structure_and_topology"],
    "alertscreening": ["alert_screening"],
    "alerts": ["alert_screening"],
    "screening": ["alert_screening"],
    "safety": ["alert_screening"],
    "structuralalerts": ["alert_screening"],
    "reactivity": ["alert_screening"],
    "reactivityandsafety": ["alert_screening"],
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


def _compute_molecular_profile(smiles: str) -> str:
    from .three_d import get_3d_properties

    return _join_sections(
        _compute_physicochemical(smiles),
        _compute_complexity(smiles),
        _compute_electronic(smiles),
        get_3d_properties(smiles, include_epsa=False),
    )


def _compute_ionization_and_solubility(smiles: str) -> str:
    return _join_sections(
        _compute_pka(smiles),
        _compute_ionization(smiles),
        _compute_logd(smiles),
        _compute_solubility(smiles),
    )


def _compute_structure_and_topology(smiles: str) -> str:
    from .functional_groups import analyze_functional_groups
    from .ring_systems import analyze_ring_systems

    return _join_sections(
        analyze_functional_groups(smiles, simple=True),
        analyze_ring_systems(smiles),
    )


def _compute_alert_screening(smiles: str) -> str:
    from .safety import screen_safety

    return screen_safety(smiles, include_smarts=False)

FEATURE_REGISTRY: Dict[str, Any] = {
    "molecular_profile": _compute_molecular_profile,
    "ionization_and_solubility": _compute_ionization_and_solubility,
    "structure_and_topology": _compute_structure_and_topology,
    "alert_screening": _compute_alert_screening,
}


def _resolve_feature_names(raw_names: List[str]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()

    for name in raw_names:
        normalized = _normalize_name(name)
        alias_targets = _FEATURE_ALIASES.get(normalized)
        if alias_targets is not None:
            for target in alias_targets:
                if target not in seen:
                    resolved.append(target)
                    seen.add(target)
            continue

        match = _fuzzy_match(name, FEATURE_NAMES)
        if match is None:
            errors.append(name)
            continue
        if match not in seen:
            resolved.append(match)
            seen.add(match)

    return resolved, errors


def _compute_features_for_smiles(smiles: str, feature_names: List[str]) -> str:
    sections = []
    for name in feature_names:
        fn = FEATURE_REGISTRY.get(name)
        if fn is None:
            continue
        try:
            result = fn(smiles)
            if result:
                sections.append(result)
        except Exception as e:
            sections.append(f"{name}: Error - {e}")
    return "\n\n".join(sections)


def get_features(smiles: str, feature_names: List[str]) -> str:
    if not feature_names:
        return f"Error: feature_names is empty. Available: {', '.join(FEATURE_NAMES)}"

    resolved, errors = _resolve_feature_names(feature_names)
    if errors:
        return (
            f"Error: Unknown feature(s): {', '.join(errors)}. "
            f"Available: {', '.join(FEATURE_NAMES)}"
        )

    return _compute_features_for_smiles(smiles, resolved)


K_NEIGHBORS = 5


def get_neighbors(
    smiles: str,
    task_name: str,
    feature_names: Optional[List[str]] = None,
    include_labels: bool = True,
) -> str:
    from .similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
        _weighted_tanimoto,
    )

    resolved_task = _resolve_task_name(task_name)
    if resolved_task is None:
        return f"Error: Unknown task '{task_name}'. Available tasks: {', '.join(TASKS)}"

    resolved_features = None
    if feature_names:
        resolved_features, errors = _resolve_feature_names(feature_names)
        if errors:
            return (
                f"Error: Unknown feature(s): {', '.join(errors)}. "
                f"Available: {', '.join(FEATURE_NAMES)}"
            )

    data = _load_task_data(resolved_task, "fingerprint")
    if data is None:
        return (
            f"Error: No precomputed fingerprint embeddings found for task '{resolved_task}'.\n"
            f"Available tasks: {', '.join(TASKS)}"
        )

    train_smiles = data["smiles"]
    train_labels = data["labels"]
    morgan_fps = data["morgan_fps"]
    feat_fps = data["feat_fps"]

    match_idx = np.where(train_smiles == smiles)[0]
    exact_mask = train_smiles == smiles

    cached_canonical_smiles = data.get("canonical_smiles")
    if len(match_idx) == 0 and cached_canonical_smiles is not None:
        query_canonical = _canonicalize_smiles(smiles)
        if query_canonical is not None:
            canonical_mask = cached_canonical_smiles == query_canonical
            canonical_match_idx = np.where(canonical_mask)[0]
            if len(canonical_match_idx) > 0:
                match_idx = canonical_match_idx
                exact_mask = exact_mask | canonical_mask

    if len(match_idx) > 0:
        query_morgan = morgan_fps[match_idx[0]]
        query_feat = feat_fps[match_idx[0]]
    else:
        query_morgan = _compute_query_fp(smiles, use_features=False)
        query_feat = _compute_query_fp(smiles, use_features=True)
        if query_morgan is None or query_feat is None:
            return f"Error: Could not compute fingerprint for query '{smiles}'."

    similarities = _weighted_tanimoto(query_morgan, query_feat, morgan_fps, feat_fps)
    similarities[exact_mask] = -np.inf

    splits = data.get("splits")
    if splits is not None:
        train_mask = splits == "train"
    else:
        split_data = _load_split_smiles(resolved_task)
        if split_data is not None:
            train_smi_set = split_data["train"]
            train_mask = np.array([s in train_smi_set for s in train_smiles])
        else:
            train_mask = np.ones(len(train_smiles), dtype=bool)

    train_sims = similarities.copy()
    train_sims[~train_mask] = -np.inf

    n_available = int((train_mask & ~exact_mask).sum())
    k = min(K_NEIGHBORS, n_available)
    top_idx = np.argsort(train_sims)[::-1][:k]

    sections = [f"Nearest Neighbors for task '{resolved_task}' (k={k}):"]

    for i, idx in enumerate(top_idx, 1):
        nbr_smiles = str(train_smiles[idx])
        sim = float(similarities[idx])
        label_part = ""
        if include_labels and train_labels is not None:
            label_part = f", label: {_label_str(int(train_labels[idx]))}"

        sections.append(f"\n{i}. {nbr_smiles} (similarity: {sim:.2f}{label_part})")

        if resolved_features:
            feat_text = _compute_features_for_smiles(nbr_smiles, resolved_features)
            if feat_text:
                indented = "\n".join("   " + line for line in feat_text.split("\n"))
                sections.append(indented)

    if k == 0:
        sections.append("No neighbors found.")

    return "\n".join(sections)


GET_FEATURES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_features",
        "description": (
            "Analyze a molecule and groups of properties"
            "such as a (1) core profile of molecular properties like LogP, TPSA, and Molecular Weight (2) pKa, ionization, logD, and solubility"
            " (3) functional groups and ring systems or (4) structural-alert screening results"
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
                    "description": (
                        "One or more evidence groups to inspect for this molecule. "
                        "molecular_profile covers molecular size, polarity, lipophilicity, hydrogen-bonding, "
                        "complexity indicators, electronic properties, and PMI-based 3D shape; ionization_and_solubility covers pKa, ionization state, "
                        "logD, and aqueous solubility; structure_and_topology covers functional groups and "
                        "ring systems; alert_screening covers structural-alert and liability-screening matches."
                    ),
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
            "Retrieve close analogs for a molecule within a specified prediction task. "
            "Use this to compare the query molecule against similar compounds, inspect "
            "their labels, and optionally review selected evidence groups for each analog."
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
                    "description": (
                        "Prediction task or assay context in which neighbors should be retrieved, "
                        "for example AMES, DILI, or hERG."
                    ),
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional evidence groups to compute for each retrieved analog, "
                        "using the same group definitions as get_features."
                    ),
                },
                "include_labels": {
                    "type": "boolean",
                    "description": "Whether to include observed task labels for the retrieved analogs. Default true.",
                },
            },
            "required": ["smiles", "task_name"],
            "additionalProperties": False,
        },
    },
}


def _task_alias(task: str) -> str:
    return TASK_ALIASES.get(task, task.lower())


def _make_task_neighbors_tool_schema(task: str) -> Dict[str, Any]:
    alias = _task_alias(task)
    return {
        "type": "function",
        "function": {
            "name": f"get_neighbors_{alias}",
            "description": (
                f"Retrieve close analogs for a molecule within the {task} prediction task. "
                "Use this to compare the query against similar compounds from the same task, "
                "inspect their labels, and optionally review selected evidence groups for each analog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Query molecule, provided as a SMILES string.",
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional evidence groups to compute for each retrieved analog, "
                            "using the same group definitions as get_features."
                        ),
                    },
                    "include_labels": {
                        "type": "boolean",
                        "description": "Whether to include observed task labels for the retrieved analogs. Default true.",
                    },
                },
                "required": ["smiles"],
                "additionalProperties": False,
            },
        },
    }


def _make_task_neighbors_callable(task: str):
    alias = _task_alias(task)

    def _get_neighbors(
        smiles: str,
        feature_names: Optional[List[str]] = None,
        include_labels: bool = True,
    ) -> str:
        return get_neighbors(
            smiles,
            task_name=task,
            feature_names=feature_names,
            include_labels=include_labels,
        )

    _get_neighbors.__name__ = f"get_neighbors_{alias}"
    return _get_neighbors


TASK_NEIGHBOR_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    task: _make_task_neighbors_tool_schema(task) for task in TASKS
}

TASK_NEIGHBOR_CALLABLES: Dict[str, Any] = {
    f"get_neighbors_{_task_alias(task)}": _make_task_neighbors_callable(task)
    for task in TASKS
}
