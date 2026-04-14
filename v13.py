"""
Version 13 therapeutic tools — v12 plus the top-20 most-cited RDKit descriptors
from the SFT trace corpus.

v13 is a strict superset of v12: every v12 feature name (and alias) keeps working,
and 20 additional flat RDKit descriptor names are added to ``FEATURE_NAMES`` so
that SFT traces narrating `Looking at BCUT2D_MRLOW = ...` can be backed by an
actual `get_features(smiles, feature_names=["BCUT2D_MRLOW", ...])` call.

Each added descriptor is computed via a single shared
`MolecularDescriptorCalculator` and rendered as one short line. Human-readable
descriptions reuse the existing
``data/sft_traces/rdkit_descriptor_descriptions.json`` file.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

from .similarity import TASKS, TASK_ALIASES
from .v11 import (
    _compute_complexity,
    _compute_electronic,
    _compute_ionization,
    _compute_logd,
    _compute_pka,
    _compute_solubility,
    _fuzzy_match,
    _label_str,
    _resolve_task_name,
)
from .v12 import (
    FEATURE_NAMES as V12_FEATURE_NAMES,
    FEATURE_REGISTRY as V12_FEATURE_REGISTRY,
    _FEATURE_ALIASES as _V12_FEATURE_ALIASES,
    _normalize_name,
)


# ---------------------------------------------------------------------------
# Top-20 RDKit descriptors from SFT trace frequency analysis (descriptors
# cited by the random-forest narratives but NOT already exposed by v12).
# Locked from the corpus scan; see plan file pure-brewing-globe.md.
# ---------------------------------------------------------------------------

TOP20_RDKIT_DESCRIPTORS: List[str] = [
    "BCUT2D_MRLOW",
    "MinPartialCharge",
    "qed",
    "VSA_EState4",
    "SMR_VSA5",
    "SMR_VSA7",
    "MaxAbsPartialCharge",
    "BCUT2D_MWHI",
    "BCUT2D_LOGPLOW",
    "MinAbsPartialCharge",
    "VSA_EState3",
    "SlogP_VSA8",
    "MaxPartialCharge",
    "BCUT2D_MRHI",
    "BCUT2D_LOGPHI",
    "FpDensityMorgan3",
    "SlogP_VSA5",
    "MinAbsEStateIndex",
    "VSA_EState7",
    "VSA_EState8",
]


# ---------------------------------------------------------------------------
# Cached descriptor calculator + descriptions
# ---------------------------------------------------------------------------

_RDKIT_DESCRIPTOR_DESCRIPTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "sft_traces", "rdkit_descriptor_descriptions.json",
)


@lru_cache(maxsize=1)
def _load_descriptor_descriptions() -> Dict[str, str]:
    try:
        with open(_RDKIT_DESCRIPTOR_DESCRIPTIONS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _get_calculator():
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    return MoleculeDescriptors.MolecularDescriptorCalculator(TOP20_RDKIT_DESCRIPTORS)


@lru_cache(maxsize=4096)
def _compute_top20_values(smiles: str) -> Dict[str, float]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: float("nan") for name in TOP20_RDKIT_DESCRIPTORS}
    vals = _get_calculator().CalcDescriptors(mol)
    return dict(zip(TOP20_RDKIT_DESCRIPTORS, vals))


def _format_descriptor_line(name: str, value: float) -> str:
    desc_map = _load_descriptor_descriptions()
    desc = desc_map.get(name, "")
    if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
        value_str = "n/a"
    elif isinstance(value, float):
        value_str = f"{value:.4f}"
    else:
        value_str = str(value)
    if desc:
        return f"{name}:\n- value: {value_str}\n- description: {desc}"
    return f"{name}:\n- value: {value_str}"


def _make_descriptor_compute_fn(name: str):
    """Build a single-feature compute fn for one RDKit descriptor."""

    def _compute(smiles: str) -> str:
        values = _compute_top20_values(smiles)
        return _format_descriptor_line(name, values.get(name, float("nan")))

    _compute.__name__ = f"_compute_{name}"
    return _compute


# ---------------------------------------------------------------------------
# Feature registry (v12 + top-20 flat RDKit descriptors)
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = list(V12_FEATURE_NAMES) + list(TOP20_RDKIT_DESCRIPTORS)

FEATURE_REGISTRY: Dict[str, Any] = {
    **V12_FEATURE_REGISTRY,
    **{name: _make_descriptor_compute_fn(name) for name in TOP20_RDKIT_DESCRIPTORS},
}

# v12 aliases keep working unchanged; we don't add new aliases for the flat
# RDKit names because the canonical descriptor name IS the user-facing one.
_FEATURE_ALIASES: Dict[str, List[str]] = dict(_V12_FEATURE_ALIASES)


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


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


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


def get_neighbors(
    smiles: str,
    task_name: str,
    feature_names: Optional[List[str]] = None,
    num_neighbors: int = 5,
    include_labels: bool = True,
) -> str:
    from .similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
        _weighted_tanimoto,
    )

    if num_neighbors <= 0:
        return "Error: num_neighbors must be a positive integer."

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
    k = min(num_neighbors, n_available)
    top_idx = np.argsort(train_sims)[::-1][:k]

    sections = [f"Nearest Neighbors for task '{resolved_task}' (k={k}):"]

    for i, idx in enumerate(top_idx, 1):
        nbr_smiles = str(train_smiles[idx])
        sim = float(similarities[idx])
        label_part = ""
        if include_labels and train_labels is not None:
            label_part = f", label: {_label_str(int(train_labels[idx]))}"

        sections.append(f"\n{i}. {nbr_smiles} (similarity: {sim:.4f}{label_part})")

        if resolved_features:
            feat_text = _compute_features_for_smiles(nbr_smiles, resolved_features)
            if feat_text:
                indented = "\n".join("   " + line for line in feat_text.split("\n"))
                sections.append(indented)

    if k == 0:
        sections.append("No neighbors found.")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# OpenAI tool schemas (mirror v12 with v13 description text)
# ---------------------------------------------------------------------------

GET_FEATURES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_features",
        "description": (
            "Compute selected molecular features for a given SMILES string. "
            "Pass a list of feature names to select which analyses to run. "
            "Supports v12 bucket categories (e.g. 'functional_groups', 'ring_systems') "
            "and individual RDKit descriptor names "
            f"(e.g. {', '.join(TOP20_RDKIT_DESCRIPTORS[:5])}, ...)."
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
                    "description": "List of feature categories or RDKit descriptor names to compute.",
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
                "num_neighbors": {
                    "type": "integer",
                    "description": "Number of nearest neighbors to return. Default 5.",
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
                    "num_neighbors": {
                        "type": "integer",
                        "description": "Number of nearest neighbors to return. Default 5.",
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
        num_neighbors: int = 5,
        include_labels: bool = True,
    ) -> str:
        return get_neighbors(
            smiles,
            task_name=task,
            feature_names=feature_names,
            num_neighbors=num_neighbors,
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
