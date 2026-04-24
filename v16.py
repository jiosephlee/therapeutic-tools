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

import json
import math
import os
import re
from functools import lru_cache
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

FEATURE_GROUP_DESCRIPTIONS: Dict[str, str] = {
    "molecular_profile": (
        "molecular size, polarity, lipophilicity, hydrogen-bonding, complexity, "
        "electronic properties, and PMI-based 3D shape"
    ),
    "ionization_and_solubility": (
        "pKa, ionization state at pH 7.4, logD, and aqueous solubility"
    ),
    "structure_and_topology": (
        "functional groups, ring systems, aromaticity, and topology summaries"
    ),
    "alert_screening": (
        "structural-alert and liability-screening categories"
    ),
}

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


def _feature_names_description_text() -> str:
    return (
        "Feature groups to include: "
        "molecular_profile, ionization_and_solubility, "
        "structure_and_topology, alert_screening."
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _format_scalar(value: Any, *, decimals: int = 2, unit: str = "") -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    if decimals == 0:
        base = str(int(round(numeric)))
    else:
        base = f"{numeric:.{decimals}f}"
    return f"{base}{unit}"


def _format_delta(neighbor_value: Any, query_value: Any, *, decimals: int = 2, unit: str = "") -> str:
    neighbor_numeric = _safe_float(neighbor_value)
    query_numeric = _safe_float(query_value)
    if neighbor_numeric is None or query_numeric is None:
        return "n/a"
    delta = neighbor_numeric - query_numeric
    if decimals == 0:
        base = str(int(round(delta)))
    else:
        base = f"{delta:.{decimals}f}"
    if delta > 0:
        base = f"+{base}"
    return f"{base}{unit}"


def _metric_neighbor_delta(
    label: str,
    query_value: Any,
    neighbor_value: Any,
    *,
    decimals: int = 2,
    unit: str = "",
) -> str:
    return (
        f"- {label}: neighbor={_format_scalar(neighbor_value, decimals=decimals, unit=unit)}, "
        f"delta={_format_delta(neighbor_value, query_value, decimals=decimals, unit=unit)}"
    )


def _load_cached_row(smiles: str) -> Optional[Dict[str, Any]]:
    from . import metadata_cache

    return metadata_cache.lookup_row(smiles)


def _cached_or_compute(cached: Optional[Dict[str, Any]], prop: str, compute_fn):
    if cached and prop in cached:
        return cached[prop]
    return compute_fn()


_SHAPE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "shape_cache.jsonl")


@lru_cache(maxsize=1)
def _load_shape_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(_SHAPE_CACHE_PATH):
        return cache
    with open(_SHAPE_CACHE_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            smi = entry.get("smiles")
            if not isinstance(smi, str):
                continue
            cache[smi] = {
                "npr1": entry.get("npr1"),
                "npr2": entry.get("npr2"),
                "shape_class": entry.get("shape_class", "unknown"),
            }
    return cache


def _lookup_shape_cache(smiles: str) -> Optional[Dict[str, Any]]:
    cache = _load_shape_cache()
    hit = cache.get(smiles)
    if hit is not None:
        return hit
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
    return cache.get(canonical)


def _compute_shape_summary(smiles: str) -> Dict[str, Any]:
    cached = _lookup_shape_cache(smiles)
    if cached is not None:
        return cached

    from rdkit.Chem import AllChem, Descriptors3D

    from .molecule_profile import _mol_from_smiles

    mol = _mol_from_smiles(smiles)
    mol = AllChem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return {"npr1": None, "npr2": None, "shape_class": "unknown"}

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
    except Exception:
        pass

    try:
        npr1 = float(Descriptors3D.NPR1(mol, confId=cid))
        npr2 = float(Descriptors3D.NPR2(mol, confId=cid))
    except Exception:
        return {"npr1": None, "npr2": None, "shape_class": "unknown"}

    if npr1 > 0.6 and npr2 > 0.6:
        shape_class = "sphere-like"
    elif npr1 < 0.3 and npr2 > 0.6:
        shape_class = "disc-like"
    elif npr1 < 0.3 and npr2 < 0.5:
        shape_class = "rod-like"
    else:
        shape_class = "intermediate"

    return {"npr1": npr1, "npr2": npr2, "shape_class": shape_class}


@lru_cache(maxsize=256)
def _compute_molecular_profile_summary(smiles: str) -> Dict[str, Any]:
    from rdkit.Chem import AllChem, Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors

    from .molecule_profile import _mol_from_smiles
    from .v14_consolidated import _f_neutral_from_pka

    cached = _load_cached_row(smiles)
    mol = _mol_from_smiles(smiles)

    molecular_weight = float(_cached_or_compute(cached, "MolWt", lambda: Descriptors.MolWt(mol)))
    heavy_atoms = int(_cached_or_compute(cached, "HeavyAtomCount", lambda: Descriptors.HeavyAtomCount(mol)))
    heteroatoms = int(_cached_or_compute(cached, "NumHeteroatoms", lambda: Lipinski.NumHeteroatoms(mol)))
    logp = float(_cached_or_compute(cached, "MolLogP", lambda: Crippen.MolLogP(mol)))
    tpsa = float(_cached_or_compute(cached, "TPSA", lambda: rdMolDescriptors.CalcTPSA(mol)))
    h_bond_donors = int(_cached_or_compute(cached, "NumHDonors", lambda: Lipinski.NumHDonors(mol)))
    h_bond_acceptors = int(_cached_or_compute(cached, "NumHAcceptors", lambda: Lipinski.NumHAcceptors(mol)))
    rotatable_bonds = int(_cached_or_compute(cached, "NumRotatableBonds", lambda: Lipinski.NumRotatableBonds(mol)))
    fraction_csp3 = float(_cached_or_compute(cached, "FractionCSP3", lambda: rdMolDescriptors.CalcFractionCSP3(mol)))
    bertz_complexity = float(_cached_or_compute(cached, "BertzCT", lambda: Descriptors.BertzCT(mol)))
    charge_max = charge_min = charge_polarization = None

    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                charge = float(atom.GetDoubleProp("_GasteigerCharge"))
                if not np.isnan(charge):
                    charges.append(charge)
            except Exception:
                pass
        if charges:
            charge_max = max(charges)
            charge_min = min(charges)
            charge_polarization = charge_max - charge_min
    except Exception:
        pass

    neutral_fraction = _safe_float(
        cached.get("f_neutral_7_4") if cached else None
    )
    if neutral_fraction is None:
        most_acidic = cached.get("most_acidic_pka") if cached else None
        most_basic = cached.get("most_basic_pka") if cached else None
        neutral_fraction = _f_neutral_from_pka(most_acidic, most_basic, 7.4)

    shape = _compute_shape_summary(smiles)

    return {
        "molecular_weight": molecular_weight,
        "heavy_atoms": heavy_atoms,
        "heteroatoms": heteroatoms,
        "logp": logp,
        "tpsa": tpsa,
        "h_bond_donors": h_bond_donors,
        "h_bond_acceptors": h_bond_acceptors,
        "rotatable_bonds": rotatable_bonds,
        "fraction_csp3": fraction_csp3,
        "bertz_complexity": bertz_complexity,
        "charge_polarization": charge_polarization,
        "neutral_fraction_7_4": neutral_fraction,
        "hall_kier_alpha": float(_cached_or_compute(cached, "HallKierAlpha", lambda: GraphDescriptors.HallKierAlpha(mol))),
        "shape": shape,
    }


@lru_cache(maxsize=256)
def _compute_ionization_and_solubility_summary(smiles: str) -> Dict[str, Any]:
    from .adme import _compact_ionization, _estimate_logd_from_pka, _get_pka_data

    cached = _load_cached_row(smiles)
    pka_data = _get_pka_data(smiles, cached)
    logd_74 = _estimate_logd_from_pka(smiles, 7.4, pka_data, cached)
    solubility_logs = _safe_float(cached.get("minimol_solubility_log_mol_L") if cached else None)
    ionization_text = _compact_ionization(smiles, ph=7.4)

    first_line = ionization_text.splitlines()[0] if ionization_text else ""
    match = re.match(r"Ionization at pH 7\.4: ([^,]+), charge (-?\d+)", first_line)
    charge_class = match.group(1) if match else "unknown"
    charge = int(match.group(2)) if match else None
    ambiguous = "Ambiguous (pKa near pH): Yes" in ionization_text

    return {
        "most_basic_pka": pka_data.get("most_basic_pka"),
        "most_acidic_pka": pka_data.get("most_acidic_pka"),
        "num_basic_sites": pka_data.get("num_basic_sites"),
        "num_acidic_sites": pka_data.get("num_acidic_sites"),
        "charge_class": charge_class,
        "charge": charge,
        "ambiguous": ambiguous,
        "logd_74": logd_74,
        "solubility_logs": solubility_logs,
    }


def _parse_functional_group_names(text: str) -> List[str]:
    names: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            names.append(line[2:].split(":")[0].strip())
    return names


@lru_cache(maxsize=256)
def _compute_structure_and_topology_summary(smiles: str) -> Dict[str, Any]:
    from rdkit.Chem import Lipinski

    from .functional_groups import analyze_functional_groups
    from .molecule_profile import _mol_from_smiles

    cached = _load_cached_row(smiles)
    mol = _mol_from_smiles(smiles)
    functional_group_text = analyze_functional_groups(smiles, simple=True)

    aromatic_atoms = int(
        _cached_or_compute(
            cached,
            "NumAromaticAtoms",
            lambda: sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
        )
    )

    return {
        "functional_groups": _parse_functional_group_names(functional_group_text),
        "ring_total": int(_cached_or_compute(cached, "RingCount", lambda: Lipinski.RingCount(mol))),
        "ring_aromatic": int(_cached_or_compute(cached, "NumAromaticRings", lambda: Lipinski.NumAromaticRings(mol))),
        "ring_aliphatic": int(_cached_or_compute(cached, "NumAliphaticRings", lambda: Lipinski.NumAliphaticRings(mol))),
        "ring_saturated": int(_cached_or_compute(cached, "NumSaturatedRings", lambda: Lipinski.NumSaturatedRings(mol))),
        "heterocycles": int(_cached_or_compute(cached, "NumHeterocycles", lambda: Lipinski.NumHeterocycles(mol))),
        "aromatic_atoms": aromatic_atoms,
        "total_atoms": int(mol.GetNumAtoms()),
    }


def _parse_alert_categories(text: str) -> List[str]:
    categories: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"\d+\.\s+(.+)", line)
        if match:
            categories.append(match.group(1).strip())
    return categories


@lru_cache(maxsize=256)
def _compute_alert_screening_summary(smiles: str) -> Dict[str, Any]:
    alert_text = _compute_alert_screening(smiles)
    categories = _parse_alert_categories(alert_text)
    return {
        "categories": categories,
        "category_count": len(categories),
    }


def _compute_feature_summary(smiles: str, feature_name: str) -> Dict[str, Any]:
    if feature_name == "molecular_profile":
        return _compute_molecular_profile_summary(smiles)
    if feature_name == "ionization_and_solubility":
        return _compute_ionization_and_solubility_summary(smiles)
    if feature_name == "structure_and_topology":
        return _compute_structure_and_topology_summary(smiles)
    if feature_name == "alert_screening":
        return _compute_alert_screening_summary(smiles)
    return {}


def _format_molecular_profile_comparison(query_summary: Dict[str, Any], neighbor_summary: Dict[str, Any]) -> str:
    shape_query = query_summary["shape"]
    shape_neighbor = neighbor_summary["shape"]
    lines = [
        "molecular_profile:",
        _metric_neighbor_delta("Molecular weight", query_summary["molecular_weight"], neighbor_summary["molecular_weight"], decimals=2, unit=" Da"),
        _metric_neighbor_delta("Heavy atoms", query_summary["heavy_atoms"], neighbor_summary["heavy_atoms"], decimals=0),
        _metric_neighbor_delta("Heteroatoms", query_summary["heteroatoms"], neighbor_summary["heteroatoms"], decimals=0),
        _metric_neighbor_delta("logP", query_summary["logp"], neighbor_summary["logp"], decimals=2),
        _metric_neighbor_delta("TPSA", query_summary["tpsa"], neighbor_summary["tpsa"], decimals=2, unit=" Å²"),
        _metric_neighbor_delta("H-bond donors", query_summary["h_bond_donors"], neighbor_summary["h_bond_donors"], decimals=0),
        _metric_neighbor_delta("H-bond acceptors", query_summary["h_bond_acceptors"], neighbor_summary["h_bond_acceptors"], decimals=0),
        _metric_neighbor_delta("Rotatable bonds", query_summary["rotatable_bonds"], neighbor_summary["rotatable_bonds"], decimals=0),
        _metric_neighbor_delta("Fsp3", query_summary["fraction_csp3"], neighbor_summary["fraction_csp3"], decimals=2),
        _metric_neighbor_delta("Bertz complexity", query_summary["bertz_complexity"], neighbor_summary["bertz_complexity"], decimals=2),
        _metric_neighbor_delta("Charge polarization", query_summary["charge_polarization"], neighbor_summary["charge_polarization"], decimals=2),
        _metric_neighbor_delta("Neutral fraction at pH 7.4", query_summary["neutral_fraction_7_4"], neighbor_summary["neutral_fraction_7_4"], decimals=4),
        _metric_neighbor_delta("PMI NPR1", shape_query["npr1"], shape_neighbor["npr1"], decimals=4),
        _metric_neighbor_delta("PMI NPR2", shape_query["npr2"], shape_neighbor["npr2"], decimals=4),
        f"- Shape class: {shape_neighbor['shape_class']}",
    ]
    return "\n".join(lines)


def _format_ionization_and_solubility_comparison(query_summary: Dict[str, Any], neighbor_summary: Dict[str, Any]) -> str:
    lines = [
        "ionization_and_solubility:",
        _metric_neighbor_delta("Strongest basic pKa", query_summary["most_basic_pka"], neighbor_summary["most_basic_pka"], decimals=2),
        _metric_neighbor_delta("Strongest acidic pKa", query_summary["most_acidic_pka"], neighbor_summary["most_acidic_pka"], decimals=2),
        _metric_neighbor_delta("Basic site count", query_summary["num_basic_sites"], neighbor_summary["num_basic_sites"], decimals=0),
        _metric_neighbor_delta("Acidic site count", query_summary["num_acidic_sites"], neighbor_summary["num_acidic_sites"], decimals=0),
        f"- Ionization state: {neighbor_summary['charge_class']} (charge {neighbor_summary['charge']})",
        f"- Ambiguous near pH 7.4: {'Yes' if neighbor_summary['ambiguous'] else 'No'}",
        _metric_neighbor_delta("LogD at pH 7.4", query_summary["logd_74"], neighbor_summary["logd_74"], decimals=2),
        _metric_neighbor_delta("Solubility logS", query_summary["solubility_logs"], neighbor_summary["solubility_logs"], decimals=2, unit=" log(mol/L)"),
    ]
    return "\n".join(lines)


def _format_structure_and_topology_comparison(query_summary: Dict[str, Any], neighbor_summary: Dict[str, Any]) -> str:
    neighbor_groups = neighbor_summary["functional_groups"]
    lines = [
        "structure_and_topology:",
        f"- Functional groups: {', '.join(neighbor_groups) if neighbor_groups else 'none'}",
        _metric_neighbor_delta("Total rings", query_summary["ring_total"], neighbor_summary["ring_total"], decimals=0),
        _metric_neighbor_delta("Aromatic rings", query_summary["ring_aromatic"], neighbor_summary["ring_aromatic"], decimals=0),
        _metric_neighbor_delta("Aliphatic rings", query_summary["ring_aliphatic"], neighbor_summary["ring_aliphatic"], decimals=0),
        _metric_neighbor_delta("Saturated rings", query_summary["ring_saturated"], neighbor_summary["ring_saturated"], decimals=0),
        _metric_neighbor_delta("Heterocycles", query_summary["heterocycles"], neighbor_summary["heterocycles"], decimals=0),
        f"- Aromatic atoms: {neighbor_summary['aromatic_atoms']}/{neighbor_summary['total_atoms']}",
    ]
    return "\n".join(lines)


def _format_alert_screening_comparison(query_summary: Dict[str, Any], neighbor_summary: Dict[str, Any]) -> str:
    neighbor_categories = neighbor_summary["categories"]
    lines = [
        "alert_screening:",
        f"- Alert categories: {', '.join(neighbor_categories) if neighbor_categories else 'none'}",
        f"- Category count: {neighbor_summary['category_count']}",
    ]
    return "\n".join(lines)


def _format_feature_comparison(
    feature_name: str,
    query_summary: Dict[str, Any],
    neighbor_summary: Dict[str, Any],
) -> str:
    if feature_name == "molecular_profile":
        return _format_molecular_profile_comparison(query_summary, neighbor_summary)
    if feature_name == "ionization_and_solubility":
        return _format_ionization_and_solubility_comparison(query_summary, neighbor_summary)
    if feature_name == "structure_and_topology":
        return _format_structure_and_topology_comparison(query_summary, neighbor_summary)
    if feature_name == "alert_screening":
        return _format_alert_screening_comparison(query_summary, neighbor_summary)
    return feature_name


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


K_NEIGHBORS = 3


def get_neighbors(
    smiles: str,
    task_name: str,
    feature_names: Optional[List[str]] = None,
) -> str:
    include_labels = True
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
    query_feature_summaries: Dict[str, Dict[str, Any]] = {}
    if resolved_features:
        for feature_name in resolved_features:
            try:
                query_feature_summaries[feature_name] = _compute_feature_summary(smiles, feature_name)
            except Exception as e:
                query_feature_summaries[feature_name] = {"error": str(e)}

    for i, idx in enumerate(top_idx, 1):
        nbr_smiles = str(train_smiles[idx])
        sim = float(similarities[idx])
        label_part = ""
        if include_labels and train_labels is not None:
            label_part = f", label: {_label_str(int(train_labels[idx]))}"

        sections.append(f"\n{i}. {nbr_smiles} (similarity: {sim:.2f}{label_part})")

        if resolved_features:
            comparison_blocks = []
            for feature_name in resolved_features:
                query_summary = query_feature_summaries.get(feature_name, {})
                if query_summary.get("error"):
                    comparison_blocks.append(
                        f"{feature_name}:\n- Error while summarizing query features: {query_summary['error']}"
                    )
                    continue
                try:
                    neighbor_summary = _compute_feature_summary(nbr_smiles, feature_name)
                    comparison_blocks.append(
                        _format_feature_comparison(feature_name, query_summary, neighbor_summary)
                    )
                except Exception as e:
                    comparison_blocks.append(
                        f"{feature_name}:\n- Error while summarizing neighbor features: {e}"
                    )
            if comparison_blocks:
                comparison_text = "\n\n".join(comparison_blocks)
                indented = "\n".join("   " + line for line in comparison_text.split("\n"))
                sections.append(indented)

    if k == 0:
        sections.append("No neighbors found.")

    return "\n".join(sections)


GET_FEATURES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_features",
        "description": (
            "Analyze a molecule and return selected evidence groups covering "
            "molecular profile, ionization and solubility, structure and topology, "
            "or structural-alert screening."
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
                    "description": _feature_names_description_text(),
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
            "Grabs similar molecules, their properties, and their actual label for the task at hand."
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
                    "description": "Feature groups to include for the similar molecules. " + _feature_names_description_text(),
                },
            },
            "required": ["smiles", "feature_names"],
            "additionalProperties": False,
        },
    },
}

TASK_NEIGHBOR_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}

TASK_NEIGHBOR_CALLABLES: Dict[str, Any] = {}
