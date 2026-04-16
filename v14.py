"""
Version 14 therapeutic tools — v12 base with metabolites removed and richer
cached scalars exposed as first-class ``get_features`` names.

Changes vs v12:
  - ``metabolites`` is no longer a valid feature name (no metabolic prediction tool).
  - Additional features (cache-backed where possible):
      * ``neutral_fraction_7_4`` — fraction of neutral microspecies at pH 7.4 from pKa
      * ``labute_asa`` — Labute surface area (``LabuteASA`` in metadata cache)
      * ``no_count`` — RDKit Lipinski NOCount (N + O heavy atoms)
      * ``num_aliphatic_carbocycles``, ``num_aromatic_carbocycles``,
        ``num_saturated_carbocycles`` — RDKit carbocycle counts

Does NOT include the v13 top-20 SFT-backed RDKit descriptors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .similarity import TASKS, TASK_ALIASES
from .v12 import (
    FEATURE_NAMES as _V12_FEATURE_NAMES,
    FEATURE_REGISTRY as _V12_FEATURE_REGISTRY,
    _FEATURE_ALIASES as _V12_FEATURE_ALIASES,
    _format_single_property,
    _normalize_name,
)
from .v11 import _fuzzy_match, _label_str, _resolve_task_name


V14_ADDITIONAL_FEATURES: List[str] = [
    "neutral_fraction_7_4",
    "labute_asa",
    "no_count",
    "num_aliphatic_carbocycles",
    "num_aromatic_carbocycles",
    "num_saturated_carbocycles",
]

FEATURE_NAMES: List[str] = [
    n for n in _V12_FEATURE_NAMES if n != "metabolites"
] + list(V14_ADDITIONAL_FEATURES)

_FEATURE_ALIASES: Dict[str, List[str]] = dict(_V12_FEATURE_ALIASES)
_FEATURE_ALIASES.update(
    {
        "neutralfraction": ["neutral_fraction_7_4"],
        "neutralfraction74": ["neutral_fraction_7_4"],
        "fneutral": ["neutral_fraction_7_4"],
        "fnneutral": ["neutral_fraction_7_4"],
        "labute": ["labute_asa"],
        "labuteasa": ["labute_asa"],
        "labutesurfacearea": ["labute_asa"],
        "nocount": ["no_count"],
        "noatomcount": ["no_count"],
    }
)


def _f_neutral_from_pka(
    most_acidic: Optional[float], most_basic: Optional[float], ph: float = 7.4
) -> float:
    """Henderson–Hasselbalch-style neutral fraction from strongest acidic/basic pKa."""
    f_neutral = 1.0
    if most_basic is not None and not (isinstance(most_basic, float) and math.isnan(most_basic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (most_basic - ph))
    if most_acidic is not None and not (isinstance(most_acidic, float) and math.isnan(most_acidic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - most_acidic))
    return min(1.0, max(1e-12, f_neutral))


def _compute_neutral_fraction_7_4(smiles: str) -> str:
    from . import metadata_cache

    cached = metadata_cache.lookup(smiles, "f_neutral_7_4")
    if cached is not None:
        val = cached
    else:
        row = metadata_cache.lookup_row(smiles)
        ma = row.get("most_acidic_pka") if row else None
        mb = row.get("most_basic_pka") if row else None
        val = _f_neutral_from_pka(ma, mb, 7.4)
    return _format_single_property(
        "Neutral fraction (pH 7.4)",
        f"Estimated fraction of neutral microspecies: {float(val):.4f}",
    )


def _compute_labute_asa(smiles: str) -> str:
    from rdkit.Chem import Descriptors

    from . import metadata_cache
    from .molecule_profile import _mol_from_smiles

    v = metadata_cache.lookup(smiles, "LabuteASA")
    if v is None:
        mol = _mol_from_smiles(smiles)
        v = float(Descriptors.LabuteASA(mol))
    return _format_single_property("Labute ASA", f"Labute surface area: {float(v):.4f}")


def _compute_no_count(smiles: str) -> str:
    from rdkit.Chem import Lipinski

    from . import metadata_cache
    from .molecule_profile import _mol_from_smiles

    v = metadata_cache.lookup(smiles, "NOCount")
    if v is None:
        mol = _mol_from_smiles(smiles)
        v = float(Lipinski.NOCount(mol))
    return _format_single_property(
        "N/O atom count",
        f"Nitrogen and oxygen heavy-atom count (RDKit NOCount): {int(v)}",
    )


def _make_carbocycle_fn(cache_col: str, title: str, line_fmt: str):
    from rdkit.Chem import Lipinski

    from . import metadata_cache
    from .molecule_profile import _mol_from_smiles

    def _fn(smiles: str) -> str:
        v = metadata_cache.lookup(smiles, cache_col)
        if v is None:
            mol = _mol_from_smiles(smiles)
            getter = getattr(Lipinski, cache_col)
            v = float(getter(mol))
        return _format_single_property(title, line_fmt.format(int(v)))

    return _fn


_V14_EXTRA_REGISTRY: Dict[str, Any] = {
    "neutral_fraction_7_4": _compute_neutral_fraction_7_4,
    "labute_asa": _compute_labute_asa,
    "no_count": _compute_no_count,
    "num_aliphatic_carbocycles": _make_carbocycle_fn(
        "NumAliphaticCarbocycles",
        "Aliphatic carbocycles",
        "Aliphatic carbocycle count: {}",
    ),
    "num_aromatic_carbocycles": _make_carbocycle_fn(
        "NumAromaticCarbocycles",
        "Aromatic carbocycles",
        "Aromatic carbocycle count: {}",
    ),
    "num_saturated_carbocycles": _make_carbocycle_fn(
        "NumSaturatedCarbocycles",
        "Saturated carbocycles",
        "Saturated carbocycle count: {}",
    ),
}

FEATURE_REGISTRY: Dict[str, Any] = {
    k: v for k, v in _V12_FEATURE_REGISTRY.items() if k != "metabolites"
}
FEATURE_REGISTRY.update(_V14_EXTRA_REGISTRY)


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
        return (
            f"Error: Unknown task '{task_name}'. "
            f"Available tasks: {', '.join(TASKS)}"
        )

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
            "Return requested molecular features for a SMILES string (v14: v12 vocabulary "
            "plus neutral fraction, Labute ASA, NOCount, carbocycle counts; metabolites removed)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Input molecule as a SMILES string.",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of feature names to return for the molecule."
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
            "Find nearest training-set neighbors for a molecule by fingerprint similarity. "
            "Returns neighbor SMILES, similarity scores, task labels, and optionally "
            "computes selected molecular features for each neighbor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Input molecule as a SMILES string.",
                },
                "task_name": {
                    "type": "string",
                    "description": "TDC task name (e.g. 'AMES', 'DILI', 'hERG').",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of feature categories or RDKit descriptor names to compute for each neighbor.",
                },
                "include_labels": {
                    "type": "boolean",
                    "description": "Whether to include task labels for neighbors. Default true.",
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
                f"Find nearest neighbors from the {task} training set with labels, "
                "similarity scores, and optionally compute selected molecular features "
                "for each neighbor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Input molecule as a SMILES string.",
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of feature categories or RDKit descriptor names to compute for each neighbor.",
                    },
                    "include_labels": {
                        "type": "boolean",
                        "description": "Whether to include task labels for neighbors. Default true.",
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
