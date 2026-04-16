"""
Version 15 therapeutic tools — TRIM-style task-bound tools.

This module adopts the two-tool surface from TRIM's
``trim.reasoning.agent_tools.openai_schemas`` exactly for the OpenAI schema
shape and field text:

  - ``get_mol_properties_and_fg``
  - ``compare_similar_mols``

Runtime implementation is adapted to this repository's existing cached
metadata, functional-group cache, and fingerprint neighbor retrieval.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .metadata_cache import lookup_row
from .similarity import TASKS
from .v11 import _label_str, _resolve_task_name


@dataclass(frozen=True)
class _DenseFeatureSpec:
    key: str
    display_name: str
    decimals: int = 2
    missing_text: str = "not available"


_DENSE_FEATURES: List[_DenseFeatureSpec] = [
    _DenseFeatureSpec("MolWt", "Molecular weight"),
    _DenseFeatureSpec("MolLogP", "logP"),
    _DenseFeatureSpec("TPSA", "TPSA"),
    _DenseFeatureSpec("MolMR", "Molar refractivity"),
    _DenseFeatureSpec("FractionCSP3", "Fraction CSP3"),
    _DenseFeatureSpec("qed", "QED"),
    _DenseFeatureSpec("HeavyAtomCount", "Heavy atom count", decimals=0),
    _DenseFeatureSpec("NumHDonors", "H-bond donor count", decimals=0),
    _DenseFeatureSpec("NumHAcceptors", "H-bond acceptor count", decimals=0),
    _DenseFeatureSpec("NumRotatableBonds", "Rotatable bond count", decimals=0),
    _DenseFeatureSpec("RingCount", "Ring count", decimals=0),
    _DenseFeatureSpec("NumAromaticRings", "Aromatic ring count", decimals=0),
    _DenseFeatureSpec("FormalCharge", "Formal charge", decimals=0),
    _DenseFeatureSpec("NumHeteroatoms", "Heteroatom count", decimals=0),
    _DenseFeatureSpec("LabuteASA", "Labute ASA"),
    _DenseFeatureSpec("MaxAbsPartialCharge", "Max abs partial charge", decimals=3),
    _DenseFeatureSpec("MinAbsPartialCharge", "Min abs partial charge", decimals=3),
    _DenseFeatureSpec("MaxEStateIndex", "Max EState index"),
    _DenseFeatureSpec("MinEStateIndex", "Min EState index"),
    _DenseFeatureSpec("NumAromaticAtoms", "Aromatic atom count", decimals=0),
    _DenseFeatureSpec("FractionAromaticAtoms", "Fraction aromatic atoms"),
    _DenseFeatureSpec("NumPositiveCharges", "Positive charge atom count", decimals=0),
    _DenseFeatureSpec("NumNegativeCharges", "Negative charge atom count", decimals=0),
    _DenseFeatureSpec("NumAliphaticRings", "Aliphatic ring count", decimals=0),
    _DenseFeatureSpec("NumSaturatedRings", "Saturated ring count", decimals=0),
    _DenseFeatureSpec("NumHeterocycles", "Heterocycle count", decimals=0),
    _DenseFeatureSpec("NumAromaticHeterocycles", "Aromatic heterocycle count", decimals=0),
    _DenseFeatureSpec("NumAliphaticHeterocycles", "Aliphatic heterocycle count", decimals=0),
    _DenseFeatureSpec("NumSaturatedHeterocycles", "Saturated heterocycle count", decimals=0),
    _DenseFeatureSpec("NumAmideBonds", "Amide bond count", decimals=0),
    _DenseFeatureSpec("BertzCT", "Bertz complexity"),
    _DenseFeatureSpec("BalabanJ", "Balaban J"),
    _DenseFeatureSpec("HallKierAlpha", "Hall-Kier alpha"),
    _DenseFeatureSpec(
        "most_basic_pka",
        "Strongest basic pKa",
        missing_text="not applicable (no basic site)",
    ),
    _DenseFeatureSpec(
        "most_acidic_pka",
        "Strongest acidic pKa",
        missing_text="not applicable (no acidic site)",
    ),
    _DenseFeatureSpec("logD_74", "logD (pH 7.4)"),
]


def _smiles_only_parameters() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES string for the molecule to analyze.",
            }
        },
        "required": ["smiles"],
        "additionalProperties": False,
    }


def _compare_tool_description(task: str | None) -> str:
    task_scope = f" for task {task}" if task else " for the current task"
    return (
        "Retrieve text-form local analog evidence"
        f"{task_scope}. The input SMILES must belong to that task dataset so the tool can identify "
        "the query split and retrieve the nearest training-set neighbors. The tool returns plain text "
        "with a short definition of query/neighbor/delta, then positive and negative neighbors "
        "(typically 3 positive and 3 negative, depending on the active manifest), each with "
        "neighbor/query/delta values for the 36 dense properties plus functional-group differences."
    )


def build_openai_agent_tool_schemas(*, task: str | None = None) -> list[dict[str, object]]:
    shared_parameters = _smiles_only_parameters()
    schemas = [
        {
            "type": "function",
            "name": "get_mol_properties_and_fg",
            "description": (
                "Return plain-text single-molecule property evidence. The tool outputs one line per "
                "dense property in 'display_name: value' form for the 36 default dense properties, "
                "followed by the molecule's present functional groups and their counts. When the "
                "strongest acidic pKa or strongest basic pKa is undefined, the text explicitly says "
                "'not applicable (no acidic/basic site)'."
            ),
            "parameters": deepcopy(shared_parameters),
            "strict": True,
        },
        {
            "type": "function",
            "name": "compare_similar_mols",
            "description": _compare_tool_description(task),
            "parameters": deepcopy(shared_parameters),
            "strict": True,
        },
    ]
    return schemas


OPENAI_AGENT_TOOL_SCHEMAS = build_openai_agent_tool_schemas()
GET_MOL_PROPERTIES_AND_FG_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])
COMPARE_SIMILAR_MOLS_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[1])
TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS: Dict[str, Dict[str, object]] = {
    task: build_openai_agent_tool_schemas(task=task)[1] for task in TASKS
}


@lru_cache(maxsize=1)
def _load_fg_cache() -> Dict[str, str]:
    import os

    cache_path = os.path.join(os.path.dirname(__file__), "cache", "fg_cache.jsonl")
    payload: Dict[str, str] = {}
    if not os.path.exists(cache_path):
        return payload
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            smiles = row.get("smiles")
            fg_text = row.get("fg")
            if isinstance(smiles, str) and isinstance(fg_text, str):
                payload[smiles] = fg_text
    return payload


def _parse_functional_group_counts(smiles: str) -> Counter[str]:
    fg_text = _load_fg_cache().get(smiles, "")
    counts: Counter[str] = Counter()
    for line in fg_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        name = stripped[2:].split(":", 1)[0].strip()
        if name:
            counts[name] += 1
    return counts


def _format_value(spec: _DenseFeatureSpec, value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return spec.missing_text
    if spec.decimals == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{spec.decimals}f}"


def _feature_values(smiles: str) -> Dict[str, Optional[float]]:
    row = lookup_row(smiles) or {}
    return {spec.key: row.get(spec.key) for spec in _DENSE_FEATURES}


def _render_dense_feature_lines(smiles: str) -> List[str]:
    values = _feature_values(smiles)
    return [
        f"{spec.display_name}: {_format_value(spec, values.get(spec.key))}"
        for spec in _DENSE_FEATURES
    ]


def _render_functional_group_lines(smiles: str) -> List[str]:
    counts = _parse_functional_group_counts(smiles)
    if not counts:
        return ["Present functional groups: none detected in cache."]
    ordered = sorted(counts.items(), key=lambda item: (item[0], item[1]))
    return [
        "Present functional groups:",
        *[f"{name}: {count}" for name, count in ordered],
    ]


def get_mol_properties_and_fg(smiles: str) -> str:
    sections = [
        * _render_dense_feature_lines(smiles),
        "",
        * _render_functional_group_lines(smiles),
    ]
    return "\n".join(sections).strip()


def _query_index_for_task(smiles: str, task: str) -> tuple[int, dict[str, Any]]:
    from .similarity import _canonicalize_smiles, _load_task_data

    data = _load_task_data(task, "fingerprint")
    if data is None:
        raise ValueError(
            f"No precomputed fingerprint embeddings found for task '{task}'. Available tasks: {', '.join(TASKS)}"
        )

    train_smiles = data["smiles"]
    match_idx = np.where(train_smiles == smiles)[0]
    if len(match_idx) > 0:
        return int(match_idx[0]), data

    cached_canonical_smiles = data.get("canonical_smiles")
    if cached_canonical_smiles is not None:
        query_canonical = _canonicalize_smiles(smiles)
        if query_canonical is not None:
            canonical_match_idx = np.where(cached_canonical_smiles == query_canonical)[0]
            if len(canonical_match_idx) > 0:
                return int(canonical_match_idx[0]), data

    raise ValueError(
        f"Input SMILES must belong to task dataset '{task}' so the query split and local neighbors can be resolved."
    )


def _load_train_mask(task: str, data: dict[str, Any]) -> np.ndarray:
    from .similarity import _load_split_smiles

    splits = data.get("splits")
    if splits is not None:
        return splits == "train"

    split_data = _load_split_smiles(task)
    if split_data is None:
        return np.ones(len(data["smiles"]), dtype=bool)

    train_set = split_data["train"]
    return np.array([s in train_set for s in data["smiles"]], dtype=bool)


def _weighted_similarity_against_task(smiles: str, task: str, data: dict[str, Any], query_idx: int) -> np.ndarray:
    from .similarity import _weighted_tanimoto

    query_morgan = data["morgan_fps"][query_idx]
    query_feat = data["feat_fps"][query_idx]
    similarities = _weighted_tanimoto(
        query_morgan,
        query_feat,
        data["morgan_fps"],
        data["feat_fps"],
    )
    similarities[query_idx] = -np.inf
    return similarities


def _safe_delta(neighbor_value: Optional[float], query_value: Optional[float]) -> Optional[float]:
    if neighbor_value is None or query_value is None:
        return None
    if np.isnan(float(neighbor_value)) or np.isnan(float(query_value)):
        return None
    return float(neighbor_value) - float(query_value)


def _render_neighbor_block(
    *,
    rank: int,
    neighbor_smiles: str,
    similarity: float,
    label: Optional[int],
    query_values: Dict[str, Optional[float]],
) -> List[str]:
    neighbor_values = _feature_values(neighbor_smiles)
    header = f"{rank}. {neighbor_smiles} | similarity={similarity:.3f}"
    if label is not None:
        header += f" | label={_label_str(int(label))}"

    lines = [header]
    for spec in _DENSE_FEATURES:
        neighbor_value = neighbor_values.get(spec.key)
        query_value = query_values.get(spec.key)
        delta = _safe_delta(neighbor_value, query_value)
        delta_text = spec.missing_text if delta is None else (
            str(int(round(delta))) if spec.decimals == 0 else f"{delta:.{spec.decimals}f}"
        )
        lines.append(
            f"  {spec.display_name}: "
            f"neighbor={_format_value(spec, neighbor_value)} | "
            f"query={_format_value(spec, query_value)} | "
            f"delta={delta_text}"
        )
    return lines


def _functional_group_difference_lines(query_smiles: str, neighbor_smiles: str) -> List[str]:
    query_counts = _parse_functional_group_counts(query_smiles)
    neighbor_counts = _parse_functional_group_counts(neighbor_smiles)
    keys = sorted(set(query_counts) | set(neighbor_counts))
    if not keys:
        return ["  Functional-group differences: none detected in cache."]

    lines = ["  Functional-group differences:"]
    any_diff = False
    for key in keys:
        q = int(query_counts.get(key, 0))
        n = int(neighbor_counts.get(key, 0))
        if q == n:
            continue
        any_diff = True
        lines.append(f"    {key}: neighbor={n} | query={q} | delta={n - q}")
    if not any_diff:
        return ["  Functional-group differences: none."]
    return lines


def _render_neighbor_group(
    *,
    heading: str,
    indices: np.ndarray,
    similarities: np.ndarray,
    smiles_array: np.ndarray,
    labels: Optional[np.ndarray],
    query_smiles: str,
    query_values: Dict[str, Optional[float]],
) -> List[str]:
    lines = [heading]
    if len(indices) == 0:
        lines.append("None.")
        return lines

    for rank, idx in enumerate(indices.tolist(), 1):
        neighbor_smiles = str(smiles_array[idx])
        neighbor_label = None if labels is None else int(labels[idx])
        lines.extend(
            _render_neighbor_block(
                rank=rank,
                neighbor_smiles=neighbor_smiles,
                similarity=float(similarities[idx]),
                label=neighbor_label,
                query_values=query_values,
            )
        )
        lines.extend(_functional_group_difference_lines(query_smiles, neighbor_smiles))
    return lines


def compare_similar_mols(smiles: str, task: Optional[str] = None) -> str:
    resolved_task = _resolve_task_name(task) if task else None
    if resolved_task is None:
        raise ValueError(
            "Could not determine task context for compare_similar_mols. "
            "This tool is task-bound and requires the active dataset task."
        )

    query_idx, data = _query_index_for_task(smiles, resolved_task)
    similarities = _weighted_similarity_against_task(smiles, resolved_task, data, query_idx)
    train_mask = _load_train_mask(resolved_task, data)
    smiles_array = data["smiles"]
    labels = data.get("labels")
    query_smiles = str(smiles_array[query_idx])
    query_values = _feature_values(query_smiles)

    candidate_mask = train_mask.copy()
    candidate_mask[query_idx] = False

    if labels is None:
        ranked = np.argsort(similarities[candidate_mask])[::-1][:6]
        candidate_indices = np.where(candidate_mask)[0][ranked]
        positives = candidate_indices[:3]
        negatives = candidate_indices[3:6]
    else:
        positive_mask = candidate_mask & (labels == 1)
        negative_mask = candidate_mask & (labels == 0)
        positives = np.where(positive_mask)[0][np.argsort(similarities[positive_mask])[::-1][:3]]
        negatives = np.where(negative_mask)[0][np.argsort(similarities[negative_mask])[::-1][:3]]

    lines = [
        f"Local analog evidence for task {resolved_task}",
        "Definitions: query = input molecule values, neighbor = retrieved training-set analog values, delta = neighbor - query.",
        f"Query SMILES: {smiles}",
        "",
        *_render_neighbor_group(
            heading="Positive neighbors:",
            indices=positives,
            similarities=similarities,
            smiles_array=smiles_array,
            labels=labels,
            query_smiles=query_smiles,
            query_values=query_values,
        ),
        "",
        *_render_neighbor_group(
            heading="Negative neighbors:",
            indices=negatives,
            similarities=similarities,
            smiles_array=smiles_array,
            labels=labels,
            query_smiles=query_smiles,
            query_values=query_values,
        ),
    ]
    return "\n".join(lines).strip()


__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "GET_MOL_PROPERTIES_AND_FG_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS",
    "build_openai_agent_tool_schemas",
    "compare_similar_mols",
    "get_mol_properties_and_fg",
]
