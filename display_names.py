"""
Shared user-facing labels for internal therapeutic-tool property names.
"""

from __future__ import annotations

import re


# Public global mapping that any tool version can import and reuse.
PROPERTY_DISPLAY_NAMES: dict[str, str] = {
    "MolWt": "Molecular weight",
    "MolLogP": "logP",
    "TPSA": "Topological polar surface area",
    "MolMR": "Molar refractivity",
    "FractionCSP3": "Fraction of sp3 carbons",
    "HeavyAtomCount": "Heavy atom count",
    "NumHDonors": "H-bond donor count",
    "NumHAcceptors": "H-bond acceptor count",
    "NumRotatableBonds": "Rotatable bond count",
    "RingCount": "Ring count",
    "NumAromaticRings": "Aromatic ring count",
    "FormalCharge": "Formal charge",
    "NumHeteroatoms": "Heteroatom count",
    "LabuteASA": "Labute surface area",
    "MaxAbsPartialCharge": "Maximum absolute partial charge",
    "MinAbsPartialCharge": "Minimum absolute partial charge",
    "MaxEStateIndex": "Maximum EState index",
    "MinEStateIndex": "Minimum EState index",
    "NumAromaticAtoms": "Aromatic atom count",
    "FractionAromaticAtoms": "Fraction of aromatic atoms",
    "NumPositiveCharges": "Positively charged atom count",
    "NumNegativeCharges": "Negatively charged atom count",
    "NumAliphaticRings": "Aliphatic ring count",
    "NumSaturatedRings": "Saturated ring count",
    "NumHeterocycles": "Heterocycle count",
    "NumAromaticHeterocycles": "Aromatic heterocycle count",
    "NumAliphaticHeterocycles": "Aliphatic heterocycle count",
    "NumSaturatedHeterocycles": "Saturated heterocycle count",
    "NumAmideBonds": "Amide bond count",
    "BertzCT": "Bertz complexity",
    "BalabanJ": "Balaban J index",
    "HallKierAlpha": "Hall-Kier alpha",
    "most_basic_pka": "Strongest basic pKa",
    "most_acidic_pka": "Strongest acidic pKa",
    "logD_74": "logD (pH 7.4)",
    "NOCount": "Nitrogen and oxygen atom count",
    "f_neutral_7_4": "Neutral fraction at pH 7.4",
    "BCUT2D_MRLOW": "Low BCUT mass-refractivity eigenvalue",
    "BCUT2D_MWHI": "High BCUT molecular-weight eigenvalue",
    "BCUT2D_LOGPLOW": "Low BCUT logP eigenvalue",
    "BCUT2D_MRHI": "High BCUT mass-refractivity eigenvalue",
    "BCUT2D_LOGPHI": "High BCUT logP eigenvalue",
    "VSA_EState3": "VSA/EState bin 3",
    "VSA_EState4": "VSA/EState bin 4",
    "VSA_EState7": "VSA/EState bin 7",
    "VSA_EState8": "VSA/EState bin 8",
    "SlogP_VSA5": "Surface-area bin 5 by logP contribution",
    "SlogP_VSA8": "Surface-area bin 8 by logP contribution",
    "SMR_VSA5": "Surface-area bin 5 by molar refractivity contribution",
    "SMR_VSA7": "Surface-area bin 7 by molar refractivity contribution",
    "FpDensityMorgan3": "Morgan fingerprint density (radius 3)",
    "MinPartialCharge": "Minimum partial charge",
    "MaxPartialCharge": "Maximum partial charge",
    "MinAbsEStateIndex": "Minimum absolute EState index",
    "qed": "QED drug-likeness score",
}


def get_display_name(name: str) -> str:
    """Return a readable user-facing label for an internal property name."""
    if name in PROPERTY_DISPLAY_NAMES:
        return PROPERTY_DISPLAY_NAMES[name]

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.replace("_", " "))
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if not spaced:
        return name
    return spaced[0].upper() + spaced[1:]


def get_semantic_display_name(name: str) -> str:
    """Public alias emphasizing that the output is the semantic display label."""
    return get_display_name(name)


__all__ = [
    "PROPERTY_DISPLAY_NAMES",
    "get_display_name",
    "get_semantic_display_name",
]
