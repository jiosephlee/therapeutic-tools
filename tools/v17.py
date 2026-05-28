"""
Version 17 therapeutic tools.

Reuses v16's ``get_features`` surface verbatim (four grouped feature buckets
over the v14_consolidated evidence) and replaces ``get_neighbors`` with an
analogue-quality block focused on signals that actually qualify a neighbor as
a structural analogue:

  - Bemis-Murcko scaffold match
  - MCS coverage
  - MMP substituent transformation (single-cut), with a small common-R-group
    lookup for human-readable labels
  - Functional-group set diff (shared / query-only / neighbor-only)
  - Size/shape physchem deltas only (MW, heavy atoms, ring count, Fsp3,
    rotatable bonds) — the physchem axes that are coarse structural-similarity
    descriptors. Property-context physchem (logP, TPSA, pKa, logD, solubility,
    charge, alerts) is intentionally dropped from this block and remains
    accessible via ``get_features``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.similarity import TASKS
from .v11 import _label_str, _resolve_task_name
from .v16 import (
    FEATURE_GROUP_DESCRIPTIONS,
    FEATURE_NAMES,
    _cached_or_compute,
    _compute_alert_screening,
    _compute_ionization_and_solubility,
    _compute_molecular_profile,
    _feature_names_description_text,
    _format_delta,
    _format_scalar,
    _join_sections,
    _load_cached_row,
    _parse_functional_group_names,
    _resolve_feature_names,
    _safe_float,
)
from ..utils.group_string_cache import lookup as _lookup_group_string


K_NEIGHBORS = 3


# ---------------------------------------------------------------------------
# Feature registry. v17 differs from v16 in two places:
#   - structure_and_topology restores the Ring Systems section (v16 format).
#   - molecular_profile enriches the Complexity Metrics "Stereocenters" line:
#     keeps the ``- Stereocenters: 0`` line when the molecule is achiral, and
#     expands to a full R/S/unspecified + double-bond breakdown when not.
# ---------------------------------------------------------------------------
def _get_stereo_breakdown(smiles: str) -> Dict[str, int]:
    """Return R / S / unspecified atom stereo + specified double-bond counts."""
    try:
        from rdkit import Chem
    except Exception:
        return {"r": 0, "s": 0, "unspec_atom": 0, "spec_bond": 0}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"r": 0, "s": 0, "unspec_atom": 0, "spec_bond": 0}
    try:
        centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    except Exception:
        centers = []
    r = sum(1 for _, label in centers if label == "R")
    s = sum(1 for _, label in centers if label == "S")
    unspec_atom = sum(1 for _, label in centers if label == "?")
    spec_bond = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        st = bond.GetStereo()
        if st in (
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS,
            Chem.BondStereo.STEREOTRANS,
        ):
            spec_bond += 1
    return {"r": r, "s": s, "unspec_atom": unspec_atom, "spec_bond": spec_bond}


def _expanded_stereo_lines(info: Dict[str, int]) -> List[str]:
    """Build the multi-line stereocenters block used when stereo is present."""
    total_atom = info["r"] + info["s"] + info["unspec_atom"]
    parts: List[str] = []
    if info["r"]:
        parts.append(f"R={info['r']}")
    if info["s"]:
        parts.append(f"S={info['s']}")
    if info["unspec_atom"]:
        parts.append(f"unspecified={info['unspec_atom']}")
    breakdown = (" (" + ", ".join(parts) + ")") if parts else ""
    lines = [f"- Stereocenters: {total_atom}{breakdown}"]
    if info["spec_bond"]:
        lines.append(
            f"- Double-bond stereo: {info['spec_bond']} specified (E/Z or cis/trans)"
        )
    return lines


def _compute_molecular_profile_v17_uncached(smiles: str) -> str:
    """v16 molecular_profile with the Stereocenters line expanded when the
    molecule has any chiral centers or stereo double bonds."""
    base = _compute_molecular_profile(smiles)
    info = _get_stereo_breakdown(smiles)
    if info["r"] + info["s"] + info["unspec_atom"] + info["spec_bond"] == 0:
        return base  # keep the plain ``- Stereocenters: 0`` line

    expanded = _expanded_stereo_lines(info)
    out_lines: List[str] = []
    replaced = False
    for ln in base.splitlines():
        if not replaced and ln.lstrip().startswith("- Stereocenters:"):
            out_lines.extend(expanded)
            replaced = True
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def _compute_molecular_profile_v17(smiles: str) -> str:
    cached = _lookup_group_string("v17_molecular_profile", smiles)
    if cached is not None:
        return cached
    return _compute_molecular_profile_v17_uncached(smiles)


def _compute_ionization_and_solubility_v17_uncached(smiles: str) -> str:
    return _compute_ionization_and_solubility(smiles)


def _compute_ionization_and_solubility_v17(smiles: str) -> str:
    cached = _lookup_group_string("v17_ionization_and_solubility", smiles)
    if cached is not None:
        return cached
    return _compute_ionization_and_solubility_v17_uncached(smiles)


def _compute_structure_and_topology_v17_uncached(smiles: str) -> str:
    from ..utils.functional_groups import analyze_functional_groups
    from ..utils.ring_systems import analyze_ring_systems

    return _join_sections(
        analyze_functional_groups(smiles, simple=True),
        analyze_ring_systems(smiles),
    )


def _compute_structure_and_topology_v17(smiles: str) -> str:
    cached = _lookup_group_string("v17_structure_and_topology", smiles)
    if cached is not None:
        return cached
    return _compute_structure_and_topology_v17_uncached(smiles)


FEATURE_REGISTRY: Dict[str, Any] = {
    "molecular_profile": _compute_molecular_profile_v17,
    "ionization_and_solubility": _compute_ionization_and_solubility_v17,
    "structure_and_topology": _compute_structure_and_topology_v17,
    "alert_screening": _compute_alert_screening,
}


def _compute_features_for_smiles(smiles: str, feature_names: List[str]) -> str:
    sections: List[str] = []
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
                    "description": _feature_names_description_text(),
                },
            },
            "required": ["smiles", "feature_names"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Common R-group lookup (canonical SMILES with [*] attachment point → label)
# ---------------------------------------------------------------------------
_COMMON_RGROUPS: Dict[str, str] = {
    # Single-cut (terminal substituent) labels: one [*] attachment.
    "*F": "-F",
    "*Cl": "-Cl",
    "*Br": "-Br",
    "*I": "-I",
    "*O": "-OH",
    "*N": "-NH2",
    "*S": "-SH",
    "*C": "-Me",
    "*CC": "-Et",
    "*CCC": "-nPr",
    "*C(C)C": "-iPr",
    "*CCCC": "-nBu",
    "*C(C)(C)C": "-tBu",
    "*C=C": "-vinyl",
    "*C#C": "-ethynyl",
    "*OC": "-OMe",
    "*OCC": "-OEt",
    "*OC(C)C": "-OiPr",
    "*NC": "-NHMe",
    "*N(C)C": "-NMe2",
    "*C(F)(F)F": "-CF3",
    "*OC(F)(F)F": "-OCF3",
    "*C#N": "-CN",
    "*[N+](=O)[O-]": "-NO2",
    "*N(=O)=O": "-NO2",
    "*C=O": "-CHO",
    "*C(C)=O": "-Ac",
    "*C(=O)O": "-COOH",
    "*C(=O)OC": "-CO2Me",
    "*C(N)=O": "-CONH2",
    "*S(N)(=O)=O": "-SO2NH2",
    "*S(C)(=O)=O": "-SO2Me",
    "*c1ccccc1": "-Ph",
    "*c1ccncc1": "-4-Py",
    "*c1ccccn1": "-2-Py",
    "*Cc1ccccc1": "-Bn",
}

_COMMON_LINKERS: Dict[str, str] = {
    # Double-cut (middle fragment) labels: two [*] attachment points.
    "*C*": "-CH2-",
    "*CC*": "-CH2CH2-",
    "*CCC*": "-CH2CH2CH2-",
    "*O*": "-O-",
    "*N*": "-NH-",
    "*S*": "-S-",
    "*OC*": "-OCH2-",
    "*CO*": "-CH2O-",
    "*C(*)=O": "-C(=O)-",
    "*NC*": "-NHCH2-",
    "*CN*": "-CH2NH-",
    "*C(*)(F)F": "-CF2-",
    "*N(*)C": "-N(Me)-",
    "*S(*)(=O)=O": "-SO2-",
    "*S(*)=O": "-S(=O)-",
    "*C(=O)N*": "-C(=O)NH-",
    "*NC(*)=O": "-NHC(=O)-",
    "*OC(*)=O": "-OC(=O)-",
    "*C(=O)O*": "-C(=O)O-",
}


def _canonicalize_rgroup(smi: str) -> Optional[str]:
    """Canonicalize an R-group SMILES fragment (with [*] attachment) via RDKit."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _rgroup_label(smi: str) -> str:
    """Return a human label for a single-attachment R-group (1-cut case)."""
    canon = _canonicalize_rgroup(smi)
    if canon is None:
        return smi
    return _COMMON_RGROUPS.get(canon, canon)


def _linker_label(smi: str) -> str:
    """Return a human label for a double-attachment linker fragment (2-cut case).

    Input has labeled dummies ``[*:1]`` / ``[*:2]``; we strip the labels to
    generic ``*`` for canonical-form lookup so that symmetric linkers
    (e.g. ``*C*``) collapse regardless of which end carries which label.
    """
    try:
        from rdkit import Chem

        normalized = smi.replace("[*:1]", "[*]").replace("[*:2]", "[*]")
        mol = Chem.MolFromSmiles(normalized)
        if mol is None:
            return smi
        canon = Chem.MolToSmiles(mol)
        return _COMMON_LINKERS.get(canon, canon)
    except Exception:
        return smi


# ---------------------------------------------------------------------------
# Scaffold / MCS / MMP helpers
# ---------------------------------------------------------------------------
def _scaffold_for(smi: str) -> Optional[str]:
    """Cache-aware Murcko scaffold SMILES lookup."""
    from . import v17_cache

    cached = v17_cache.lookup(smi)
    if cached is not None and "murcko_scaffold" in cached:
        return cached["murcko_scaffold"] or None
    from ..utils.similarity import _get_murcko_scaffold

    return _get_murcko_scaffold(smi)


def _scaffold_match(query_smi: str, neighbor_smi: str) -> Tuple[bool, Optional[str]]:
    """Return (same_scaffold, neighbor_scaffold_smi_if_different)."""
    q_scaf = _scaffold_for(query_smi)
    n_scaf = _scaffold_for(neighbor_smi)
    if q_scaf is None or n_scaf is None:
        return False, n_scaf
    return q_scaf == n_scaf, n_scaf


def _strip_dummy_maps(smi: str) -> str:
    return smi.replace("[*:1]", "[*]").replace("[*:2]", "[*]")


def _swap_dummy_labels(smi: str) -> str:
    """Swap atom-map labels :1 ↔ :2 on dummy atoms."""
    return (
        smi.replace("[*:1]", "[*:\x00]")
        .replace("[*:2]", "[*:1]")
        .replace("[*:\x00]", "[*:2]")
    )


def _canon(smi: str) -> Optional[str]:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _heavy_atom_count(smi: str) -> int:
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() != 0)
    except Exception:
        return 0


def compute_fragmentations(smi: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Compute 1-cut and 2-cut MMP fragmentation maps for a molecule.

    Returns (single_cut_map, double_cut_map).
      - single_cut_map: canonical(context, no atom maps) → canonical(R-group, no atom maps)
      - double_cut_map: canonical(context with [*:1]/[*:2]) → canonical(variable with [*:1]/[*:2]).
        Both label orderings of each fragmentation are stored so intersection
        is insensitive to how each molecule happened to assign :1 vs :2.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMMPA

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {}, {}
        try:
            frags = rdMMPA.FragmentMol(
                mol, minCuts=1, maxCuts=2, maxCutBonds=30, resultsAsMols=False
            )
        except Exception:
            return {}, {}
    except Exception:
        return {}, {}

    single: Dict[str, str] = {}
    double: Dict[str, str] = {}
    for pair in frags:
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        core_smi, chain_smi = pair
        if not core_smi:
            # 1-cut: chain_smi is "piece1[*:1].piece2[*:1]"
            if not chain_smi or "." not in chain_smi:
                continue
            pieces = chain_smi.split(".")
            if len(pieces) != 2:
                continue
            a, b = pieces
            if _heavy_atom_count(a) >= _heavy_atom_count(b):
                ctx, var = a, b
            else:
                ctx, var = b, a
            ctx_canon = _canon(_strip_dummy_maps(ctx))
            var_canon = _canon(_strip_dummy_maps(var))
            if ctx_canon and var_canon:
                single.setdefault(ctx_canon, var_canon)
        else:
            # 2-cut: core_smi is the variable fragment, chain_smi is the context.
            if not chain_smi:
                continue
            ctx1 = _canon(chain_smi)
            var1 = _canon(core_smi)
            if ctx1 and var1:
                double.setdefault(ctx1, var1)
            ctx2 = _canon(_swap_dummy_labels(chain_smi))
            var2 = _canon(_swap_dummy_labels(core_smi))
            if ctx2 and var2 and ctx2 != ctx1:
                double.setdefault(ctx2, var2)
    return single, double


def _fragmentations_for(smi: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Cache-aware fragmentation lookup."""
    from . import v17_cache

    cached = v17_cache.lookup(smi)
    if cached is not None and "mmp_single" in cached and "mmp_double" in cached:
        return cached["mmp_single"], cached["mmp_double"]
    return compute_fragmentations(smi)


def _mmp_transformation(query_smi: str, neighbor_smi: str) -> Optional[str]:
    """Find a 1- or 2-cut MMP between query and neighbor (Hussain-Rea style).

    Uses the v17 per-molecule cache for fragmentation maps when available.
    Single-cut matches are preferred over double-cut when both exist
    (simpler, more interpretable output).
    """
    try:
        q_single, q_double = _fragmentations_for(query_smi)
        n_single, n_double = _fragmentations_for(neighbor_smi)

        # 1. Prefer single-cut hits — more interpretable terminal-substituent swap.
        shared_single = [
            ctx for ctx in (set(q_single) & set(n_single)) if q_single[ctx] != n_single[ctx]
        ]
        if shared_single:
            best_ctx = max(shared_single, key=_heavy_atom_count)
            return f"{_rgroup_label(q_single[best_ctx])} → {_rgroup_label(n_single[best_ctx])}"

        # 2. Fall back to double-cut hits (linker / middle-fragment swap).
        shared_double = [
            ctx for ctx in (set(q_double) & set(n_double)) if q_double[ctx] != n_double[ctx]
        ]
        if shared_double:
            best_ctx = max(shared_double, key=_heavy_atom_count)
            return (
                f"linker: {_linker_label(q_double[best_ctx])} → "
                f"{_linker_label(n_double[best_ctx])}"
            )

        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Functional-group diff
# ---------------------------------------------------------------------------
def _functional_group_set_uncached(smiles: str) -> List[str]:
    try:
        from ..utils.functional_groups import analyze_functional_groups

        text = analyze_functional_groups(smiles, simple=True)
        return _parse_functional_group_names(text)
    except Exception:
        return []


def _functional_group_set(smiles: str) -> List[str]:
    from . import v17_cache

    cached = v17_cache.lookup(smiles)
    if cached is not None and "functional_groups" in cached:
        return list(cached["functional_groups"])
    return _functional_group_set_uncached(smiles)


_FG_COUNT_RE = __import__("re").compile(r"^(.*?)(?:\s*\(x(\d+)\))?$")


def _fg_counts(names: List[str]) -> Dict[str, int]:
    """Parse FG names like 'benzene' or 'benzene (x2)' into {name: count}."""
    counts: Dict[str, int] = {}
    for raw in names:
        m = _FG_COUNT_RE.match(raw.strip())
        if not m:
            continue
        name = m.group(1).strip()
        if not name:
            continue
        n = int(m.group(2)) if m.group(2) else 1
        counts[name] = counts.get(name, 0) + n
    return counts


def _format_fg_diff(query_smi: str, neighbor_smi: str) -> str:
    q_counts = _fg_counts(_functional_group_set(query_smi))
    n_counts = _fg_counts(_functional_group_set(neighbor_smi))

    shared: Dict[str, int] = {}
    query_only: Dict[str, Tuple[int, bool]] = {}  # (delta, is_extra_copy)
    neighbor_only: Dict[str, Tuple[int, bool]] = {}
    for name in set(q_counts) | set(n_counts):
        q = q_counts.get(name, 0)
        n = n_counts.get(name, 0)
        shared_count = min(q, n)
        if shared_count > 0:
            shared[name] = shared_count
        if q > n:
            query_only[name] = (q - n, n > 0)
        elif n > q:
            neighbor_only[name] = (n - q, q > 0)

    def _fmt_shared(d: Dict[str, int]) -> str:
        if not d:
            return "{}"
        parts = [name if c == 1 else f"{name} (x{c})" for name, c in sorted(d.items())]
        return "{" + ", ".join(parts) + "}"

    def _fmt_only(d: Dict[str, Tuple[int, bool]]) -> str:
        if not d:
            return "{}"
        parts: List[str] = []
        for name, (delta, is_extra) in sorted(d.items()):
            if is_extra:
                parts.append(f"{delta} extra {name}")
            elif delta == 1:
                parts.append(name)
            else:
                parts.append(f"{name} (x{delta})")
        return "{" + ", ".join(parts) + "}"

    return (
        "Functional-group diff:\n"
        f"     - shared:        {_fmt_shared(shared)}\n"
        f"     - query-only:    {_fmt_only(query_only)}\n"
        f"     - neighbor-only: {_fmt_only(neighbor_only)}"
    )


# ---------------------------------------------------------------------------
# Size/shape summary (only structural-similarity-adjacent physchem axes)
# ---------------------------------------------------------------------------
def _compute_size_shape_summary(smiles: str) -> Dict[str, Any]:
    from . import v17_cache

    cached_entry = v17_cache.lookup(smiles)
    if cached_entry is not None:
        ss = cached_entry.get("size_shape")
        if ss:
            return dict(ss)
    return _compute_size_shape_summary_uncached(smiles)


def _compute_size_shape_summary_uncached(smiles: str) -> Dict[str, Any]:
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    from ..utils.molecule_profile import _mol_from_smiles

    cached = _load_cached_row(smiles)
    mol = _mol_from_smiles(smiles)

    return {
        "molecular_weight": float(
            _cached_or_compute(cached, "MolWt", lambda: Descriptors.MolWt(mol))
        ),
        "heavy_atoms": int(
            _cached_or_compute(
                cached, "HeavyAtomCount", lambda: Descriptors.HeavyAtomCount(mol)
            )
        ),
        "ring_total": int(
            _cached_or_compute(cached, "RingCount", lambda: Lipinski.RingCount(mol))
        ),
        "fraction_csp3": float(
            _cached_or_compute(
                cached, "FractionCSP3", lambda: rdMolDescriptors.CalcFractionCSP3(mol)
            )
        ),
        "rotatable_bonds": int(
            _cached_or_compute(
                cached, "NumRotatableBonds", lambda: Lipinski.NumRotatableBonds(mol)
            )
        ),
    }


def _format_size_shape_deltas(query: Dict[str, Any], neighbor: Dict[str, Any]) -> str:
    lines = [
        "Size/shape deltas:",
        f"     - Molecular weight: neighbor={_format_scalar(neighbor['molecular_weight'], decimals=2, unit=' Da')}, "
        f"delta={_format_delta(neighbor['molecular_weight'], query['molecular_weight'], decimals=2, unit=' Da')}",
        f"     - Heavy atoms:      neighbor={_format_scalar(neighbor['heavy_atoms'], decimals=0)}, "
        f"delta={_format_delta(neighbor['heavy_atoms'], query['heavy_atoms'], decimals=0)}",
        f"     - Total rings:      neighbor={_format_scalar(neighbor['ring_total'], decimals=0)}, "
        f"delta={_format_delta(neighbor['ring_total'], query['ring_total'], decimals=0)}",
        f"     - Fsp3:             neighbor={_format_scalar(neighbor['fraction_csp3'], decimals=2)}, "
        f"delta={_format_delta(neighbor['fraction_csp3'], query['fraction_csp3'], decimals=2)}",
        f"     - Rotatable bonds:  neighbor={_format_scalar(neighbor['rotatable_bonds'], decimals=0)}, "
        f"delta={_format_delta(neighbor['rotatable_bonds'], query['rotatable_bonds'], decimals=0)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-neighbor analogue-quality block
# ---------------------------------------------------------------------------
def _format_analogue_block(
    query_smi: str,
    neighbor_smi: str,
    query_size_shape: Dict[str, Any],
    similarity: float,
    label_str: Optional[str],
    index: int,
) -> str:
    from ..utils.similarity import _compute_mcs_summary

    label_part = f", label: {label_str}" if label_str is not None else ""
    lines: List[str] = [
        f"\n{index}. {neighbor_smi} (similarity: {similarity:.2f}{label_part})"
    ]

    # Scaffold match
    try:
        same_scaf, neighbor_scaf = _scaffold_match(query_smi, neighbor_smi)
        if same_scaf:
            lines.append("   Scaffold match: same Bemis-Murcko scaffold")
        else:
            scaf_part = f" (neighbor scaffold: {neighbor_scaf})" if neighbor_scaf else ""
            lines.append(f"   Scaffold match: different{scaf_part}")
    except Exception as e:
        lines.append(f"   Scaffold match: error ({e})")

    # MCS
    try:
        mcs = _compute_mcs_summary(query_smi, neighbor_smi)
        lines.append(f"   MCS: {mcs if mcs else 'none found'}")
    except Exception as e:
        lines.append(f"   MCS: error ({e})")

    # MMP transformation
    try:
        mmp = _mmp_transformation(query_smi, neighbor_smi)
        if mmp:
            lines.append(f"   MMP transformation: {mmp}")
        else:
            lines.append("   MMP transformation: no 1- or 2-cut MMP (structurally too divergent)")
    except Exception as e:
        lines.append(f"   MMP transformation: error ({e})")

    # Functional-group diff
    try:
        lines.append("   " + _format_fg_diff(query_smi, neighbor_smi))
    except Exception as e:
        lines.append(f"   Functional-group diff: error ({e})")

    # Size/shape deltas
    try:
        nbr_size_shape = _compute_size_shape_summary(neighbor_smi)
        lines.append("   " + _format_size_shape_deltas(query_size_shape, nbr_size_shape))
    except Exception as e:
        lines.append(f"   Size/shape deltas: error ({e})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_neighbors(smiles: str, task_name: str) -> str:
    from ..utils.similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
        _weighted_tanimoto,
    )

    resolved_task = _resolve_task_name(task_name)
    if resolved_task is None:
        return f"Error: Unknown task '{task_name}'. Available tasks: {', '.join(TASKS)}"

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

    sections: List[str] = [f"Nearest Neighbors for task '{resolved_task}' (k={k}):"]

    if k == 0:
        sections.append("No neighbors found.")
        return "\n".join(sections)

    try:
        query_size_shape = _compute_size_shape_summary(smiles)
    except Exception as e:
        return f"Error computing size/shape summary for query '{smiles}': {e}"

    for i, idx in enumerate(top_idx, 1):
        nbr_smiles = str(train_smiles[idx])
        sim = float(similarities[idx])
        label_str = _label_str(int(train_labels[idx])) if train_labels is not None else None
        sections.append(
            _format_analogue_block(smiles, nbr_smiles, query_size_shape, sim, label_str, i)
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
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
    "TASK_NEIGHBOR_CALLABLES",
    "TASK_NEIGHBOR_TOOL_SCHEMAS",
    "get_features",
    "get_neighbors",
]
