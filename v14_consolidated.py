"""
Version 14 consolidated therapeutic tools.

This variant restores the v11-style consolidated feature buckets while keeping
the v14 removal of ``metabolites``. The ``physicochemical`` bucket also folds
in the additional cached scalar properties introduced in v14.
"""

from __future__ import annotations

import math
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


FEATURE_NAMES: List[str] = [
    "physicochemical",
    "complexity",
    "electronic",
    "functional_groups",
    "ring_systems",
    "3d_properties",
    "structural_alerts",
    "pka",
    "ionization",
    "logd",
    "solubility",
]


def _f_neutral_from_pka(
    most_acidic: Optional[float], most_basic: Optional[float], ph: float = 7.4
) -> float:
    """Henderson-Hasselbalch-style neutral fraction from strongest acidic/basic pKa."""
    f_neutral = 1.0
    if most_basic is not None and not (isinstance(most_basic, float) and math.isnan(most_basic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (most_basic - ph))
    if most_acidic is not None and not (isinstance(most_acidic, float) and math.isnan(most_acidic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - most_acidic))
    return min(1.0, max(1e-12, f_neutral))


def _resolve_feature_names(raw_names: List[str]) -> tuple[list[str], list[str]]:
    resolved = []
    errors = []
    for name in raw_names:
        match = _fuzzy_match(name, FEATURE_NAMES)
        if match is not None:
            resolved.append(match)
        else:
            errors.append(name)
    return resolved, errors


def _compute_physicochemical(smiles: str) -> str:
    """Grouped v11-style physicochemical section with the v14 scalar additions."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    from . import metadata_cache
    from .molecule_profile import _mol_from_smiles

    cached = metadata_cache.lookup_row(smiles)

    def _c(prop: str, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    mol = _mol_from_smiles(smiles)

    mw = float(_c("MolWt", lambda: Descriptors.MolWt(mol)))
    heavy_atoms = int(_c("HeavyAtomCount", lambda: Descriptors.HeavyAtomCount(mol)))
    heteroatoms = int(_c("NumHeteroatoms", lambda: Lipinski.NumHeteroatoms(mol)))
    logp = float(_c("MolLogP", lambda: Crippen.MolLogP(mol)))
    tpsa = float(_c("TPSA", lambda: rdMolDescriptors.CalcTPSA(mol)))
    hbd = int(_c("NumHDonors", lambda: Lipinski.NumHDonors(mol)))
    hba = int(_c("NumHAcceptors", lambda: Lipinski.NumHAcceptors(mol)))
    rotatable = int(_c("NumRotatableBonds", lambda: Lipinski.NumRotatableBonds(mol)))
    fsp3 = float(_c("FractionCSP3", lambda: rdMolDescriptors.CalcFractionCSP3(mol)))
    mr = float(_c("MolMR", lambda: Crippen.MolMR(mol)))
    labute_asa = float(_c("LabuteASA", lambda: Descriptors.LabuteASA(mol)))
    no_count = int(_c("NOCount", lambda: Lipinski.NOCount(mol)))
    aliphatic_carbocycles = int(
        _c("NumAliphaticCarbocycles", lambda: Lipinski.NumAliphaticCarbocycles(mol))
    )
    aromatic_carbocycles = int(
        _c("NumAromaticCarbocycles", lambda: Lipinski.NumAromaticCarbocycles(mol))
    )
    saturated_carbocycles = int(
        _c("NumSaturatedCarbocycles", lambda: Lipinski.NumSaturatedCarbocycles(mol))
    )

    neutral_fraction = metadata_cache.lookup(smiles, "f_neutral_7_4")
    if neutral_fraction is None:
        most_acidic = cached.get("most_acidic_pka") if cached else None
        most_basic = cached.get("most_basic_pka") if cached else None
        neutral_fraction = _f_neutral_from_pka(most_acidic, most_basic, 7.4)

    return "\n".join(
        [
            "Physicochemical Properties:",
            f"- Molecular weight: {mw:.2f} Da",
            f"- Heavy atoms: {heavy_atoms}, Heteroatoms: {heteroatoms}",
            f"- logP (Wildman-Crippen): {logp:.2f}",
            f"- TPSA: {tpsa:.2f} Å²",
            f"- H-bond donors: {hbd}, H-bond acceptors: {hba}",
            f"- Rotatable bonds: {rotatable}",
            f"- Fraction sp3 carbons (Fsp3): {fsp3:.2f}",
            f"- Molar refractivity: {mr:.2f}",
            f"- Neutral fraction at pH 7.4: {float(neutral_fraction):.4f}",
            f"- Labute surface area: {labute_asa:.4f}",
            f"- Nitrogen and oxygen atom count: {no_count}",
            (
                "- Carbocycle counts: "
                f"aliphatic={aliphatic_carbocycles}, "
                f"aromatic={aromatic_carbocycles}, "
                f"saturated={saturated_carbocycles}"
            ),
        ]
    )


FEATURE_REGISTRY: Dict[str, Any] = {
    "physicochemical": _compute_physicochemical,
    "complexity": _compute_complexity,
    "electronic": _compute_electronic,
    "functional_groups": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.functional_groups", fromlist=["analyze_functional_groups"]
    ).analyze_functional_groups(s, simple=True),
    "ring_systems": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.ring_systems", fromlist=["analyze_ring_systems"]
    ).analyze_ring_systems(s),
    "3d_properties": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.three_d", fromlist=["get_3d_properties"]
    ).get_3d_properties(s, include_epsa=False),
    "structural_alerts": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.safety", fromlist=["screen_safety"]
    ).screen_safety(s),
    "pka": _compute_pka,
    "ionization": _compute_ionization,
    "logd": _compute_logd,
    "solubility": _compute_solubility,
}


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
            "Analyze a molecule and return selected groups of medicinal-chemistry evidence, "
            "such as physicochemical properties; pKa, ionization, logD, and solubility; "
            "functional groups and ring systems; electronic and structural-alert liabilities; "
            "or 3D shape and flexibility features."
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
                        "One or more evidence groups to inspect for this molecule: "
                        "physicochemical_properties (size, polarity, lipophilicity, hydrogen-bonding), "
                        "complexity_and_topology (stereochemistry, ring complexity, scaffold complexity), "
                        "pka_and_ionization (acid/base character, protonation state), "
                        "distribution_and_solubility (logD and aqueous solubility), "
                        "functional_groups_and_rings (functional-group motifs and ring systems), "
                        "reactivity_and_alerts (electronic/reactivity signals and structural alerts), "
                        "or shape_and_flexibility (3D shape, conformer-related geometry, flexibility)."
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
                    "description": "Prediction task or assay context, for example AMES, DILI, or hERG.",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional evidence groups to compute for each retrieved analog, using the same "
                        "group definitions as get_features."
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
                            "Optional evidence groups to compute for each retrieved analog, using the same "
                            "group definitions as get_features."
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
