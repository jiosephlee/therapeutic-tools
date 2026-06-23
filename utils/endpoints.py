"""Canonical low-level endpoint functions for therapeutic chemistry features.

Each endpoint returns typed data by default (``format="raw"``) and concise text
with ``format="text"``. Semantic modules should compose these functions instead
of reimplementing individual descriptor lookups.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED as QEDModule, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import metadata_cache
from .structured_feature_cache import get_or_compute as _structured_cache_get_or_compute

EndpointFormat = Literal["raw", "text"]


def _mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _invalid_text(name: str, smiles: str) -> str:
    return f"{name}: invalid SMILES {smiles!r}"


def _fmt_number(name: str, value: Optional[float], unit: str = "", decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{name}: unavailable"
    suffix = f" {unit}" if unit else ""
    return f"{name}: {value:.{decimals}f}{suffix}"


def _cached(smiles: str, prop: str) -> Optional[float]:
    return metadata_cache.lookup(smiles, prop)


def _numeric_endpoint(
    smiles: str,
    prop: str,
    compute,
    label: str,
    *,
    unit: str = "",
    decimals: int = 2,
    format: EndpointFormat = "raw",
) -> float | str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text(label, smiles)

    def _compute_value() -> float:
        cached = _cached(smiles, prop)
        return float(cached) if cached is not None else float(compute(mol))

    value = float(_structured_cache_get_or_compute(prop, smiles, None, _compute_value))
    if format == "raw":
        return value
    return _fmt_number(label, value, unit=unit, decimals=decimals)


def mol_weight(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    return _numeric_endpoint(smiles, "MolWt", Descriptors.MolWt, "Molecular weight", unit="Da", format=format)


def logp(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    return _numeric_endpoint(smiles, "MolLogP", Crippen.MolLogP, "logP", format=format)


def tpsa(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    return _numeric_endpoint(smiles, "TPSA", rdMolDescriptors.CalcTPSA, "TPSA", unit="A^2", format=format)


def qed(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    return _numeric_endpoint(smiles, "qed", QEDModule.qed, "QED", format=format)


def solubility_log_s(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    def _compute_value() -> float:
        cached = _cached(smiles, "minimol_solubility_log_mol_L")
        if cached is not None:
            return float(cached)
        from .solubility import _esol

        return float(_esol(smiles))

    value = float(_structured_cache_get_or_compute("solubility_log_s", smiles, None, _compute_value))
    if math.isnan(value):
        return None if format == "raw" else _invalid_text("Solubility logS", smiles)
    if format == "raw":
        return value
    return _fmt_number("Solubility logS", value, unit="log mol/L")


def pka_summary(smiles: str, format: EndpointFormat = "raw") -> dict[str, Any] | str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text("pKa summary", smiles)

    def _compute_value() -> dict[str, Any]:
        cached = metadata_cache.lookup_row(smiles)
        try:
            from .adme import _get_pka_data

            return _get_pka_data(smiles, cached)
        except Exception:
            return {
                "most_acidic_pka": cached.get("most_acidic_pka") if cached else None,
                "most_basic_pka": cached.get("most_basic_pka") if cached else None,
                "num_acidic_sites": int(cached.get("num_acidic_sites", 0)) if cached else 0,
                "num_basic_sites": int(cached.get("num_basic_sites", 0)) if cached else 0,
                "acid_sites": None,
                "base_sites": None,
            }

    data = _structured_cache_get_or_compute("pka_summary", smiles, None, _compute_value)
    if format == "raw":
        return data
    acid = data.get("most_acidic_pka")
    base = data.get("most_basic_pka")
    parts = []
    parts.append("most acidic pKa n/a" if acid is None else f"most acidic pKa {float(acid):.2f}")
    parts.append("most basic pKa n/a" if base is None else f"most basic pKa {float(base):.2f}")
    return "pKa summary: " + ", ".join(parts)


def logd_74(smiles: str, format: EndpointFormat = "raw") -> float | str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text("LogD at pH 7.4", smiles)

    def _compute_value() -> float:
        cached = _cached(smiles, "logD_74")
        if cached is not None:
            return float(cached)
        try:
            from .adme import _estimate_logd_from_pka

            return float(_estimate_logd_from_pka(smiles, 7.4, pka_summary(smiles), metadata_cache.lookup_row(smiles)))
        except Exception:
            return float(Crippen.MolLogP(mol))

    value = float(_structured_cache_get_or_compute("logd_74", smiles, {"ph": 7.4}, _compute_value))
    if format == "raw":
        return value
    return _fmt_number("LogD at pH 7.4", value)


_FG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("carboxylic acid", "C(=O)[OX2H1]"),
    ("carboxylate", "C(=O)[O-]"),
    ("carboxylic ester", "C(=O)O[#6]"),
    ("amide", "C(=O)N"),
    ("ketone", "[#6]C(=O)[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("alcohol", "[OX2H][#6]"),
    ("phenol", "c[OX2H]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("primary amine", "[NX3;H2][#6]"),
    ("secondary amine", "[NX3;H1]([#6])[#6]"),
    ("tertiary amine", "[NX3;H0]([#6])([#6])[#6]"),
    ("nitrile", "C#N"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("halide", "[F,Cl,Br,I]"),
    ("alkene", "C=C"),
    ("alkyne", "C#C"),
    ("benzene", "c1ccccc1"),
)


def functional_group_names(smiles: str, format: EndpointFormat = "raw") -> list[str] | str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text("Functional groups", smiles)

    def _compute_value() -> list[str]:
        names: list[str] = []
        for name, smarts in _FG_PATTERNS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                names.append(name)
        return names

    names = _structured_cache_get_or_compute("functional_group_names", smiles, None, _compute_value)
    if format == "raw":
        return names
    return "Functional groups: " + (", ".join(names) if names else "none detected")


def ring_summary(smiles: str, format: EndpointFormat = "raw") -> dict[str, Any] | str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text("Ring summary", smiles)

    def _compute_value() -> dict[str, Any]:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        return {
            "ring_count": int(_cached(smiles, "RingCount") or Lipinski.RingCount(mol)),
            "aromatic_rings": int(_cached(smiles, "NumAromaticRings") or Lipinski.NumAromaticRings(mol)),
            "aliphatic_rings": int(_cached(smiles, "NumAliphaticRings") or Lipinski.NumAliphaticRings(mol)),
            "saturated_rings": int(_cached(smiles, "NumSaturatedRings") or Lipinski.NumSaturatedRings(mol)),
            "heterocycles": int(_cached(smiles, "NumHeterocycles") or Lipinski.NumHeterocycles(mol)),
            "largest_ring_size": max((len(ring) for ring in atom_rings), default=0),
        }

    summary = _structured_cache_get_or_compute("ring_summary", smiles, None, _compute_value)
    if format == "raw":
        return summary
    return (
        "Ring summary: "
        f"{summary['ring_count']} rings, {summary['aromatic_rings']} aromatic, "
        f"{summary['heterocycles']} heterocycles"
    )


def murcko_scaffold(smiles: str, format: EndpointFormat = "raw") -> str | None:
    mol = _mol(smiles)
    if mol is None:
        return None if format == "raw" else _invalid_text("Murcko scaffold", smiles)

    def _compute_value() -> str:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core) if core.GetNumAtoms() > 0 else ""

    scaffold = str(_structured_cache_get_or_compute("murcko_scaffold", smiles, None, _compute_value))
    if format == "raw":
        return scaffold
    return f"Murcko scaffold: {scaffold if scaffold else '(acyclic molecule - no scaffold)'}"


__all__ = [
    "functional_group_names",
    "logd_74",
    "logp",
    "mol_weight",
    "murcko_scaffold",
    "pka_summary",
    "qed",
    "ring_summary",
    "solubility_log_s",
    "tpsa",
]
