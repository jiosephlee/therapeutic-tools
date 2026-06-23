"""Render therapeutic feature bundles from typed primitive values."""

from __future__ import annotations

import math
from typing import Any


def fmt_float(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}{suffix}"


def render_molecule_profile(values: dict[str, Any]) -> str:
    stereo = values.get("stereocenter_breakdown") or {}
    stereo_total = int(values.get("NumAtomStereoCenters") or 0)
    stereo_line = f"- Stereocenters: {stereo_total}"
    if stereo_total:
        parts = []
        if stereo.get("r"):
            parts.append(f"R={stereo['r']}")
        if stereo.get("s"):
            parts.append(f"S={stereo['s']}")
        if stereo.get("unspecified"):
            parts.append(f"unspecified={stereo['unspecified']}")
        stereo_line = f"- Stereocenters: {stereo_total} ({', '.join(parts)})" if parts else stereo_line
    return "\n".join(
        [
            "Physicochemical Properties:",
            f"- Molecular weight: {fmt_float(values.get('MolWt'), 2, ' Da')}",
            f"- Heavy atoms: {values.get('HeavyAtomCount')}, Heteroatoms: {values.get('NumHeteroatoms')}",
            f"- logP (Wildman-Crippen): {fmt_float(values.get('MolLogP'))}",
            f"- TPSA: {fmt_float(values.get('TPSA'), 2, ' A^2')}",
            f"- H-bond donors: {values.get('NumHDonors')}, H-bond acceptors: {values.get('NumHAcceptors')}",
            f"- Rotatable bonds: {values.get('NumRotatableBonds')}",
            f"- Fraction sp3 carbons (Fsp3): {fmt_float(values.get('FractionCSP3'))}",
            f"- Molar refractivity: {fmt_float(values.get('MolMR'))}",
            f"- Neutral fraction at pH 7.4: {fmt_float(values.get('neutral_fraction_7_4'), 4)}",
            f"- Labute surface area: {fmt_float(values.get('LabuteASA'), 4)}",
            f"- Nitrogen and oxygen atom count: {values.get('NOCount')}",
            (
                "- Carbocycle counts: "
                f"aliphatic={values.get('NumAliphaticCarbocycles')}, "
                f"aromatic={values.get('NumAromaticCarbocycles')}, "
                f"saturated={values.get('NumSaturatedCarbocycles')}"
            ),
            "",
            "Complexity Metrics:",
            f"- Bertz complexity: {fmt_float(values.get('BertzCT'))}",
            f"- Balaban J: {fmt_float(values.get('BalabanJ'))}",
            f"- Hall-Kier alpha: {fmt_float(values.get('HallKierAlpha'))}",
            (
                "- Kappa shape indices: "
                f"K1={fmt_float(values.get('Kappa1'))}, "
                f"K2={fmt_float(values.get('Kappa2'))}, "
                f"K3={fmt_float(values.get('Kappa3'))}"
            ),
            stereo_line,
            "",
            "Electronic Properties:",
            f"- Charge polarization: {fmt_float(values.get('charge_polarization'))}",
            f"- Max absolute partial charge: {fmt_float(values.get('MaxAbsPartialCharge'))}",
            f"- Min absolute partial charge: {fmt_float(values.get('MinAbsPartialCharge'))}",
        ]
    )


def render_ionization_and_solubility(values: dict[str, Any]) -> str:
    pka_text = str(values.get("pka_ionization_logd_text") or "").strip()
    log_s = values.get("minimol_solubility_log_s")
    solubility = f"Solubility: logS = {fmt_float(log_s)} log(mol/L)"
    return "\n\n".join(part for part in (pka_text, solubility) if part)


def render_structure_and_topology(values: dict[str, Any]) -> str:
    groups = values.get("functional_groups") or []
    if not isinstance(groups, list):
        groups = []
    return "\n".join(
        [
            "Functional Groups:",
            f"- {', '.join(str(group) for group in groups) if groups else 'none detected'}",
            "",
            "Ring Systems:",
            f"- Total rings: {values.get('ring_count')}",
            f"- Aromatic rings: {values.get('aromatic_rings')}",
            f"- Aliphatic rings: {values.get('aliphatic_rings')}",
            f"- Saturated rings: {values.get('saturated_rings')}",
            f"- Heterocycles: {values.get('heterocycles')}",
            f"- Largest ring size: {values.get('largest_ring_size')}",
        ]
    )


def _ionization_summary(ionization_text: Any) -> str | None:
    """Pull the compact '<state>, charge <n>' clause from the verbose ionization text."""
    text = str(ionization_text or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    prefix = "Ionization at pH 7.4:"
    if first.startswith(prefix):
        return first[len(prefix):].strip() or None
    return first or None


def render_concise_profile_v18(values: dict[str, Any]) -> str:
    """Concise, one-feature-per-line physicochemical block (v18).

    A curated ~16-field subset of the v17 bundles, reusing the exact same
    primitives (no new compute): the size/lipophilicity/polarity/H-bond,
    ionization-at-pH-7.4, solubility, and pharmacophore/ring descriptors that
    actually drive whether assay behaviour transfers between two molecules. The
    collinear size proxies and graph-theoretic complexity indices from
    ``molecular_profile`` are intentionally dropped. Built for the assay-transfer
    prompt, where token budget is tight.
    """
    lines = ["Molecular profile (concise):"]
    lines.append(f"- Molecular weight: {fmt_float(values.get('MolWt'), 2, ' Da')}")
    lines.append(f"- logP (Wildman-Crippen): {fmt_float(values.get('MolLogP'))}")
    lines.append(f"- logD (pH 7.4): {fmt_float(values.get('logD_74'))}")
    lines.append(f"- TPSA: {fmt_float(values.get('TPSA'), 2, ' A^2')}")
    lines.append(f"- H-bond donors: {values.get('NumHDonors')}")
    lines.append(f"- H-bond acceptors: {values.get('NumHAcceptors')}")
    lines.append(f"- Rotatable bonds: {values.get('NumRotatableBonds')}")
    lines.append(f"- Fraction sp3 carbons (Fsp3): {fmt_float(values.get('FractionCSP3'))}")

    ion = _ionization_summary(values.get("ionization_text"))
    if ion:
        lines.append(f"- Ionization (pH 7.4): {ion}")
    pka = values.get("pka_summary")
    if isinstance(pka, dict):
        if pka.get("most_acidic_pka") is not None:
            lines.append(f"- Strongest acidic pKa: {fmt_float(pka.get('most_acidic_pka'))}")
        if pka.get("most_basic_pka") is not None:
            lines.append(f"- Strongest basic pKa: {fmt_float(pka.get('most_basic_pka'))}")
    lines.append(
        f"- Aqueous solubility (logS): {fmt_float(values.get('minimol_solubility_log_s'))} log(mol/L)"
    )

    groups = values.get("functional_groups") or []
    if not isinstance(groups, list):
        groups = []
    lines.append(
        f"- Functional groups: {', '.join(str(g) for g in groups) if groups else 'none detected'}"
    )
    lines.append(f"- Aromatic rings: {values.get('aromatic_rings')}")
    lines.append(f"- Aliphatic rings: {values.get('aliphatic_rings')}")
    lines.append(f"- Heterocycles: {values.get('heterocycles')}")
    lines.append(f"- Stereocenters: {values.get('NumAtomStereoCenters')}")
    return "\n".join(lines)


def render_alert_screening(values: dict[str, Any]) -> str:
    categories = values.get("alert_categories") or []
    if not categories:
        return "Structural Alerts:\nNo structural alerts found."
    return "\n".join(["Structural Alerts:", *(f"- {category}" for category in categories)])


def render_bundle(bundle_name: str, values: dict[str, Any]) -> str:
    if bundle_name == "molecule_profile":
        return render_molecule_profile(values)
    if bundle_name == "ionization_and_solubility":
        return render_ionization_and_solubility(values)
    if bundle_name == "structure_and_topology":
        return render_structure_and_topology(values)
    if bundle_name == "concise_profile":
        return render_concise_profile_v18(values)
    if bundle_name == "alert_screening":
        return render_alert_screening(values)
    raise KeyError(f"unknown therapeutic bundle {bundle_name!r}")

