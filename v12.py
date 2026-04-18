"""
Version 12 therapeutic tools — granular get_features + task-specific neighbors.

This version keeps the v11 structure but replaces the single
``physicochemical`` feature bucket with individual property-level features.
Legacy ``physicochemical`` requests are still accepted as a shorthand alias
that expands to the individual properties.
"""

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


PHYSICOCHEMICAL_FEATURES: List[str] = [
    "molecular_weight",
    "heavy_atom_count",
    "heteroatom_count",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "fraction_csp3",
    "molar_refractivity",
]

FEATURE_NAMES: List[str] = PHYSICOCHEMICAL_FEATURES + [
    "complexity",
    "electronic",
    "functional_groups",
    "ring_systems",
    "3d_properties",
    "structural_alerts",
    "metabolites",
    "pka",
    "ionization",
    "logd",
    "solubility",
]

_FEATURE_ALIASES: Dict[str, List[str]] = {
    "physicochemical": PHYSICOCHEMICAL_FEATURES,
    "physicochem": PHYSICOCHEMICAL_FEATURES,
    "mw": ["molecular_weight"],
    "molecularweight": ["molecular_weight"],
    "heavyatoms": ["heavy_atom_count"],
    "heavyatomcount": ["heavy_atom_count"],
    "heteroatoms": ["heteroatom_count"],
    "heteroatomcount": ["heteroatom_count"],
    "clogp": ["logp"],
    "logp": ["logp"],
    "logd74proxy": ["logd"],
    "polarsurfacearea": ["tpsa"],
    "hydrogenbonddonors": ["h_bond_donors"],
    "hbonddonors": ["h_bond_donors"],
    "hbd": ["h_bond_donors"],
    "hydrogenbondacceptors": ["h_bond_acceptors"],
    "hbondacceptors": ["h_bond_acceptors"],
    "hba": ["h_bond_acceptors"],
    "numrotatablebonds": ["rotatable_bonds"],
    "fsp3": ["fraction_csp3"],
    "fractionsp3": ["fraction_csp3"],
    "molarrefraction": ["molar_refractivity"],
    "mr": ["molar_refractivity"],
}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _format_single_property(title: str, text: str) -> str:
    return f"{title}:\n- {text}"


@lru_cache(maxsize=4096)
def _get_physicochemical_values(smiles: str) -> Dict[str, float]:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    from . import metadata_cache
    from .molecule_profile import _mol_from_smiles

    cached = metadata_cache.lookup_row(smiles)

    def _c(prop: str, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    mol = _mol_from_smiles(smiles)

    return {
        "molecular_weight": float(_c("MolWt", lambda: Descriptors.MolWt(mol))),
        "heavy_atom_count": int(_c("HeavyAtomCount", lambda: Descriptors.HeavyAtomCount(mol))),
        "heteroatom_count": int(_c("NumHeteroatoms", lambda: Lipinski.NumHeteroatoms(mol))),
        "logp": float(_c("MolLogP", lambda: Crippen.MolLogP(mol))),
        "tpsa": float(_c("TPSA", lambda: rdMolDescriptors.CalcTPSA(mol))),
        "h_bond_donors": int(_c("NumHDonors", lambda: Lipinski.NumHDonors(mol))),
        "h_bond_acceptors": int(_c("NumHAcceptors", lambda: Lipinski.NumHAcceptors(mol))),
        "rotatable_bonds": int(_c("NumRotatableBonds", lambda: Lipinski.NumRotatableBonds(mol))),
        "fraction_csp3": float(_c("FractionCSP3", lambda: rdMolDescriptors.CalcFractionCSP3(mol))),
        "molar_refractivity": float(_c("MolMR", lambda: Crippen.MolMR(mol))),
    }


def _compute_molecular_weight(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Molecular Weight", f"Molecular weight: {values['molecular_weight']:.2f} Da")


def _compute_heavy_atom_count(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Heavy Atom Count", f"Heavy atoms: {values['heavy_atom_count']}")


def _compute_heteroatom_count(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Heteroatom Count", f"Heteroatoms: {values['heteroatom_count']}")


def _compute_logp_property(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("logP", f"logP (Wildman-Crippen): {values['logp']:.2f}")


def _compute_tpsa_property(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("TPSA", f"TPSA: {values['tpsa']:.2f} Å²")


def _compute_h_bond_donors(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("H-Bond Donors", f"H-bond donors: {values['h_bond_donors']}")


def _compute_h_bond_acceptors(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("H-Bond Acceptors", f"H-bond acceptors: {values['h_bond_acceptors']}")


def _compute_rotatable_bonds(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Rotatable Bonds", f"Rotatable bonds: {values['rotatable_bonds']}")


def _compute_fraction_csp3(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Fraction CSP3", f"Fraction sp3 carbons (Fsp3): {values['fraction_csp3']:.2f}")


def _compute_molar_refractivity(smiles: str) -> str:
    values = _get_physicochemical_values(smiles)
    return _format_single_property("Molar Refractivity", f"Molar refractivity: {values['molar_refractivity']:.2f}")


FEATURE_REGISTRY: Dict[str, Any] = {
    "molecular_weight": _compute_molecular_weight,
    "heavy_atom_count": _compute_heavy_atom_count,
    "heteroatom_count": _compute_heteroatom_count,
    "logp": _compute_logp_property,
    "tpsa": _compute_tpsa_property,
    "h_bond_donors": _compute_h_bond_donors,
    "h_bond_acceptors": _compute_h_bond_acceptors,
    "rotatable_bonds": _compute_rotatable_bonds,
    "fraction_csp3": _compute_fraction_csp3,
    "molar_refractivity": _compute_molar_refractivity,
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
    "metabolites": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.metabolism", fromlist=["predict_metabolites"]
    ).predict_metabolites(s, max_metabolites=1),
    "pka": _compute_pka,
    "ionization": _compute_ionization,
    "logd": _compute_logd,
    "solubility": _compute_solubility,
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


K_NEIGHBORS = 3


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
            "Compute selected molecular features for a given SMILES string. "
            "Pass a list of feature names to select which analyses to run."
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
                    "description": "List of feature categories to compute.",
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
            "their labels, and optionally review selected feature groups for each analog."
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
                    "description": "Optional feature groups to compute for each retrieved analog.",
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
                "inspect their labels, and optionally review selected feature groups for each analog."
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
                        "description": "Optional feature groups to compute for each retrieved analog.",
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
