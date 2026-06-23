"""Primitive and bundle registry for DuckDB-first therapeutic features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MINIMOL_METHOD = "MiniMol + Ridge (AqSolDB)"
DEFAULT_PARAMS_JSON = "{}"

BUNDLE_NAMES = (
    "molecule_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
    "concise_profile",
)

PUBLIC_FEATURE_NAMES = (
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
)

MOLECULE_PROFILE_PRIMITIVES = (
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "MolMR",
    "qed",
    "LabuteASA",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAromaticCarbocycles",
    "NumSaturatedCarbocycles",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "NumAtomStereoCenters",
    "NumUnspecifiedAtomStereoCenters",
    "stereocenter_breakdown",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "charge_polarization",
    "NumAromaticAtoms",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeterocycles",
    "neutral_fraction_7_4",
)
PKA_PRIMITIVES = ("pka_summary", "ionization_text", "logD_74", "pka_ionization_logd_text")
MINIMOL_PRIMITIVES = ("minimol_solubility_log_s",)
STRUCTURE_PRIMITIVES = (
    "functional_groups",
    "ring_count",
    "aromatic_rings",
    "aliphatic_rings",
    "saturated_rings",
    "heterocycles",
    "largest_ring_size",
)
ALERT_PRIMITIVES = ("alert_categories", "alert_category_count")

# Curated subset spanning the molecule_profile, pka_ionization_logd, minimol, and
# structure families — the transfer-relevant descriptors only. Every name here is
# already a registered primitive, so the v18 concise bundle reuses the existing
# primitive cache with no new compute.
CONCISE_PROFILE_PRIMITIVES = (
    "MolWt",
    "MolLogP",
    "logD_74",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "ionization_text",
    "pka_summary",
    "minimol_solubility_log_s",
    "functional_groups",
    "aromatic_rings",
    "aliphatic_rings",
    "heterocycles",
    "NumAtomStereoCenters",
)


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    compute_family: str
    runtime_name: str
    value_type: str
    method: str


def _build_specs() -> dict[str, PrimitiveSpec]:
    specs: dict[str, PrimitiveSpec] = {}
    for name in MOLECULE_PROFILE_PRIMITIVES:
        specs[name] = PrimitiveSpec(name, "molecule_profile", "openrlhf", "json", "RDKit descriptors")
    for name in PKA_PRIMITIVES:
        specs[name] = PrimitiveSpec(name, "pka_ionization_logd", "openrlhf", "json", "MolGpKa/RDKit")
    for name in MINIMOL_PRIMITIVES:
        specs[name] = PrimitiveSpec(name, "minimol_solubility", "minimol", "number", MINIMOL_METHOD)
    for name in STRUCTURE_PRIMITIVES:
        specs[name] = PrimitiveSpec(name, "structure_and_topology", "openrlhf", "json", "RDKit SMARTS/ring topology")
    for name in ALERT_PRIMITIVES:
        specs[name] = PrimitiveSpec(name, "alert_screening", "openrlhf", "json", "RDKit structural alerts")
    return specs


PRIMITIVE_SPECS = _build_specs()

BUNDLE_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "molecule_profile": MOLECULE_PROFILE_PRIMITIVES,
    "ionization_and_solubility": PKA_PRIMITIVES + MINIMOL_PRIMITIVES,
    "structure_and_topology": STRUCTURE_PRIMITIVES,
    "alert_screening": ALERT_PRIMITIVES,
    "concise_profile": CONCISE_PROFILE_PRIMITIVES,
}

FEATURE_GROUP_DESCRIPTIONS: dict[str, str] = {
    "molecular_profile": (
        "molecular size, polarity, lipophilicity, hydrogen-bonding, complexity, "
        "electronic properties, and PMI-based 3D shape"
    ),
    "ionization_and_solubility": "pKa, ionization state at pH 7.4, logD, and aqueous solubility",
    "structure_and_topology": "functional groups, ring systems, aromaticity, and topology summaries",
    "alert_screening": "structural-alert and liability-screening categories",
    "concise_profile": (
        "concise transfer-relevant physicochemical block: size, lipophilicity "
        "(logP/logD), polarity, H-bonding, ionization at pH 7.4, solubility, "
        "functional groups, ring systems, and stereocenters"
    ),
}

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "molecularprofile": ("molecular_profile",),
    "moleculeprofile": ("molecular_profile",),
    "profile": ("molecular_profile",),
    "molecularproperties": ("molecular_profile",),
    "core": ("molecular_profile",),
    "coreproperties": ("molecular_profile",),
    "physicochemical": ("molecular_profile",),
    "complexity": ("molecular_profile",),
    "electronic": ("molecular_profile",),
    "shape": ("molecular_profile",),
    "flexibility": ("molecular_profile",),
    "3d": ("molecular_profile",),
    "3dproperties": ("molecular_profile",),
    "shapeandflexibility": ("molecular_profile",),
    "developability": ("molecular_profile", "ionization_and_solubility"),
    "ionization": ("ionization_and_solubility",),
    "pka": ("ionization_and_solubility",),
    "logd": ("ionization_and_solubility",),
    "solubility": ("ionization_and_solubility",),
    "ionizationandsolubility": ("ionization_and_solubility",),
    "structure": ("structure_and_topology",),
    "topology": ("structure_and_topology",),
    "functionalgroups": ("structure_and_topology",),
    "ringsystems": ("structure_and_topology",),
    "structureandtopology": ("structure_and_topology",),
    "alertscreening": ("alert_screening",),
    "alerts": ("alert_screening",),
    "screening": ("alert_screening",),
    "safety": ("alert_screening",),
    "structuralalerts": ("alert_screening",),
    "reactivity": ("alert_screening",),
    "reactivityandsafety": ("alert_screening",),
    "conciseprofile": ("concise_profile",),
    "concise": ("concise_profile",),
    "conciseprofilev18": ("concise_profile",),
    "v18": ("concise_profile",),
}

BUNDLE_ALIASES: dict[str, str] = {
    "molecular_profile": "molecule_profile",
    "molecule_profile": "molecule_profile",
    "ionization_and_solubility": "ionization_and_solubility",
    "structure_and_topology": "structure_and_topology",
    "alert_screening": "alert_screening",
    "concise_profile": "concise_profile",
    "all_features": "all_features",
}


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _levenshtein(s: str, t: str) -> int:
    if len(s) < len(t):
        s, t = t, s
    if not t:
        return len(s)
    previous = list(range(len(t) + 1))
    for i, cs in enumerate(s, 1):
        current = [i]
        for j, ct in enumerate(t, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (cs != ct),
                )
            )
        previous = current
    return previous[-1]


def _fuzzy_match(query: str, candidates: Iterable[str], max_dist: int = 2) -> str | None:
    normalized_query = normalize_name(query)
    best: tuple[int, str] | None = None
    for candidate in candidates:
        dist = _levenshtein(normalized_query, normalize_name(candidate))
        if best is None or dist < best[0]:
            best = (dist, candidate)
    return best[1] if best is not None and best[0] <= max_dist else None


def resolve_feature_names(raw_names: Iterable[str]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        name = str(raw_name)
        alias_targets = FEATURE_ALIASES.get(normalize_name(name))
        if alias_targets is None and name in PUBLIC_FEATURE_NAMES:
            alias_targets = (name,)
        if alias_targets is None:
            match = _fuzzy_match(name, PUBLIC_FEATURE_NAMES)
            if match is None:
                errors.append(name)
                continue
            alias_targets = (match,)
        for target in alias_targets:
            if target not in seen:
                resolved.append(target)
                seen.add(target)
    return resolved, errors


def normalize_bundle_name(bundle_name: str) -> str:
    return BUNDLE_ALIASES.get(str(bundle_name), str(bundle_name))


def normalize_feature_name(bundle_name: str) -> str:
    normalized = normalize_bundle_name(bundle_name)
    return "molecular_profile" if normalized == "molecule_profile" else normalized


def feature_set_key(feature_names: Iterable[str]) -> str:
    return ",".join(normalize_bundle_name(name) for name in feature_names)


def primitives_for_bundle(bundle_name: str) -> tuple[str, ...]:
    normalized = normalize_bundle_name(bundle_name)
    try:
        return BUNDLE_PRIMITIVES[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown therapeutic bundle {bundle_name!r}") from exc


def validate_primitives(primitive_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(name) for name in primitive_names))
    unknown = [name for name in names if name not in PRIMITIVE_SPECS]
    if unknown:
        raise KeyError(f"unknown therapeutic primitive(s): {', '.join(sorted(unknown))}")
    return names


def compute_families_for_primitives(primitive_names: Iterable[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for name in validate_primitives(primitive_names):
        family = PRIMITIVE_SPECS[name].compute_family
        families.setdefault(family, []).append(name)
    return families


def runtime_families(families: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    family_runtime = {
        "molecule_profile": "openrlhf",
        "pka_ionization_logd": "openrlhf",
        "structure_and_topology": "openrlhf",
        "alert_screening": "openrlhf",
        "minimol_solubility": "minimol",
    }
    for family in families:
        runtime_name = family_runtime[family]
        grouped.setdefault(runtime_name, []).append(family)
    return grouped


def feature_names_description_text() -> str:
    return (
        "Feature groups to include: "
        "molecular_profile, ionization_and_solubility, "
        "structure_and_topology, alert_screening."
    )

