"""
Tool 6: Toxicophore Screening — semantic structural alert classification.

Groups raw RDKit filter catalog matches into mechanistically meaningful
toxicophore categories, each with a one-line note explaining why it matters.

Also screens against ToxAlerts (OCHEM) endpoint-specific SMARTS patterns
to provide direct task-relevant toxicity predictions.
"""

import os
import csv
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache


# ── Semantic toxicophore categories ─────────────────────────────────────
#
# Each category: (display_name, mechanistic_note, list_of_patterns)
# Patterns are matched case-insensitively against raw alert descriptions.
# Order matters — first match wins (an alert is assigned to one category).

_TOXICOPHORE_CATEGORIES: List[Tuple[str, str, List[str]]] = [
    (
        "Michael acceptor",
        "Electrophilic alkene conjugated with electron-withdrawing group; "
        "can covalently modify cysteine residues on proteins. "
        "Relevant to: skin sensitization, mutagenicity, hepatotoxicity.",
        ["michael", "alpha beta-unsaturated", "acrylate", "acrylonitrile",
         "vinyl_sulphone", "vinyl sulphone", "maleimide", "ene_one",
         "ene_quin_methide", "vinyl michael", "alkynyl michael",
         "trisub_bis_act_olefin"],
    ),
    (
        "Alkylating agent",
        "Can transfer alkyl groups to DNA bases, causing mutations. "
        "Relevant to: mutagenicity (AMES), carcinogenicity.",
        ["mustard", "nitrogen_mustard", "alkyl_halide", "alkyl halide",
         "r1 reactive alkyl halide", "allyl_halide", "benzyl_halide",
         "halo_olefin", "halo_acrylate", "beta halo carbonyl",
         "alpha_halo_carbonyl", "alpha halo carbonyl", "alpha_halo_ewg",
         "primary_halide_sulfate", "secondary_halide_sulfate",
         "tertiary_halide_sulfate", "halogenated_ring", "halogenated ring",
         "filter26_alkyl_halide", "filter30_beta_halo_carbonyl",
         "filter75_alkyl_br_i", "filter3_allyl_halide",
         "filter45_allyl_halide", "filter4_alpha_halo_carbonyl",
         "n-c-hal", "halo_imino"],
    ),
    (
        "Epoxide / aziridine / thioepoxide",
        "Strained 3-membered ring electrophile; reacts with DNA and proteins. "
        "Relevant to: mutagenicity, carcinogenicity (e.g. diol-epoxide metabolites of PAHs).",
        ["epoxide", "aziridine", "thioepoxide", "three-membered_heterocycle",
         "three-membered heterocycle", "three_membered_heterocycle",
         "filter40_epoxide_aziridine", "i6 epoxide"],
    ),
    (
        "Aldehyde",
        "Reactive carbonyl that forms Schiff bases with lysine residues on proteins. "
        "Relevant to: skin sensitization, mutagenicity.",
        ["aldehyde", "filter38_aldehyde", "azoalkanal"],
    ),
    (
        "Nitroaromatic / nitro group",
        "Can be reduced to reactive nitroso/hydroxylamine intermediates that damage DNA. "
        "Relevant to: mutagenicity (AMES), carcinogenicity.",
        ["nitro", "nitroso", "nitrosamine", "n-nitroso", "filter11_nitrosamin",
         "filter12_nitroso", "oxygen-nitrogen single bond",
         "oxygen-nitrogen_single_bond", "trinitro", "dinitrobenzene",
         "nitro aromatic", "aromatic no2"],
    ),
    (
        "Polycyclic aromatic hydrocarbon (PAH)",
        "Planar polyaromatic system; metabolized by CYP1A1/1B1 to diol-epoxides that bind DNA. "
        "Relevant to: carcinogenicity, AhR activation (Tox21-NR-AhR).",
        ["polycyclic_aromatic", "polycyclic aromatic", "polynuclear_aromatic",
         "linear_polycyclic_aromatic", "pyrene", "phenanthrene", "phenalene",
         "filter63_polyaromatic", "filter68_anthracene", "branched_polycyclic",
         "multiple aromatic rings", "pah"],
    ),
    (
        "Quinone",
        "Redox-active; generates reactive oxygen species and can deplete glutathione. "
        "Relevant to: hepatotoxicity (DILI), oxidative stress, mutagenicity.",
        ["quinone", "chinone", "hydroquinone", "filter23_ortho_quinone",
         "filter53_para_quinone", "disulfonylimino", "ortho_hydroimino",
         "para_hydroimino"],
    ),
    (
        "Acyl halide / acid anhydride",
        "Highly reactive acylating agents; react with nucleophilic residues non-selectively. "
        "Relevant to: general toxicity, skin sensitization.",
        ["acid_halide", "acid halide", "acyl_halide", "acyl halide",
         "anhydride", "acid_anhydride", "filter2_acyl", "filter27_anhydride",
         "sulfonyl_halide", "sulfonyl halide", "sufonyl halide",
         "filter25_sulfonyl_halide", "carbonyl_halide"],
    ),
    (
        "Hydrazine / hydrazone / azide",
        "Can be metabolically activated to reactive radicals; genotoxic. "
        "Relevant to: mutagenicity, hepatotoxicity.",
        ["hydrazine", "hydrazone", "hydrazide", "acylhydrazide", "azide",
         "azido", "diazo", "diazonium", "filter20_hydrazine", "filter7_diazo",
         "carbazide", "any carbazide", "hzone_", "hzide_"],
    ),
    (
        "Isocyanate / isothiocyanate",
        "Electrophilic; reacts with nucleophilic amino acids (cysteine, lysine). "
        "Relevant to: skin sensitization, respiratory sensitization.",
        ["isocyanate", "isothiocyanate", "carbodiimide", "isonitrile",
         "filter8_thio_isocyanat"],
    ),
    (
        "Thiol / disulfide",
        "Reactive sulfur; can disrupt disulfide bonds in proteins or generate ROS. "
        "Relevant to: protein interference (PAINS), redox cycling.",
        ["thiol", "disulfide", "disulphide", "polysulfide", "filter56_ss_bond",
         "thioles_(not_aromatic)"],
    ),
    (
        "Acyl cyanide / sulfonyl cyanide",
        "Highly electrophilic carbon attacked by biological nucleophiles. "
        "Relevant to: general toxicity, reactivity.",
        ["acyl_cyanide", "acyl cyanide", "sulfonyl_cyanide", "sulfonyl cyanide",
         "cyanophosphonate", "cyanohydrin", "filter21_cyanhydrin",
         "cyanamide"],
    ),
    (
        "Azo compound",
        "Azo bonds can be reductively cleaved to release aromatic amines (potential carcinogens). "
        "Relevant to: carcinogenicity, mutagenicity.",
        ["azo_a(", "azo_aryl", "azo_amino", "azo_filter", "azo group",
         "azo_group", "azobenzene", "azocyanamide", "filter5_azo",
         "p-aminoaryl_diazo", "dye "],
    ),
    (
        "Polyphenol / catechol",
        "Redox-active; auto-oxidizes to quinones and generates ROS. "
        "Relevant to: assay interference (PAINS), oxidative stress.",
        ["catechol", "polyphenol", "dihydroxybenzene", "trihydroxyphenyl",
         "filter57_polyphenol", "filter58_polyphenol", "hydroquin_a"],
    ),
    (
        "Heavy metal / organometallic",
        "Metals can inhibit enzymes by binding to active-site residues. "
        "Relevant to: general toxicity, enzyme inhibition.",
        ["heavy_metal", "heavy metal", "contains_metal", "filter9_metal",
         "metal_carbon_bond", "unacceptable atoms"],
    ),
    (
        "Peroxide / oxime",
        "Oxidizing agents; can generate free radicals. "
        "Relevant to: general toxicity, instability.",
        ["peroxide", "oxime", "hydroxamic_acid", "hydroxamic acid",
         "hydroxamate", "filter32_oo_bond", "filter18_oxime_ester",
         "triacyloxime"],
    ),
]

# Pre-compile patterns for speed
_COMPILED_CATEGORIES = [
    (name, note, [p.lower() for p in pats])
    for name, note, pats in _TOXICOPHORE_CATEGORIES
]


def _classify_alert(description: str) -> str | None:
    """Map a raw alert description to a semantic category name, or None."""
    desc_lower = description.strip().lower().replace("_", " ")
    for cat_name, _, patterns in _COMPILED_CATEGORIES:
        for pat in patterns:
            if pat.replace("_", " ") in desc_lower:
                return cat_name
    return None


# ── ToxAlerts (OCHEM) endpoint-specific screening ─────────────────────
#
# Maps ToxAlerts PROPERTY values to TDC task-relevant endpoint groups.

_TOXALERTS_ENDPOINT_MAP = {
    "Skin sensitization": (
        "Skin sensitization",
        "Structural alerts for skin sensitization via haptenation "
        "(covalent binding to skin proteins). "
        "Relevant to: Skin_Reaction."
    ),
    "Genotoxic carcinogenicity, mutagenicity": (
        "Genotoxic carcinogenicity / mutagenicity",
        "DNA-reactive motifs that cause mutations via direct genotoxic mechanisms. "
        "Relevant to: AMES, Carcinogens_Lagunin."
    ),
    "Non-genotoxic carcinogenicity": (
        "Non-genotoxic carcinogenicity",
        "Promotes cancer through non-DNA-reactive mechanisms "
        "(receptor activation, epigenetic changes, chronic inflammation). "
        "Relevant to: Carcinogens_Lagunin, ClinTox."
    ),
    "Idiosyncratic toxicity (RM formation)": (
        "Reactive metabolite formation (idiosyncratic toxicity)",
        "Can form reactive metabolites via CYP metabolism that covalently "
        "modify liver proteins, triggering immune-mediated hepatotoxicity. "
        "Relevant to: DILI."
    ),
    "Reactive, unstable, toxic": (
        "Reactive / unstable / toxic groups",
        "Generally reactive or unstable functional groups flagged across "
        "multiple toxicity endpoints. "
        "Relevant to: AMES, DILI, ClinTox."
    ),
    "Potential electrophilic agents": (
        "Electrophilic agents",
        "Electrophilic centers that can react with biological nucleophiles "
        "(protein thiols, DNA bases). "
        "Relevant to: Skin_Reaction, AMES."
    ),
    "Chelating agents": (
        "Chelating agents",
        "Can sequester metal ions essential for enzyme function; "
        "may interfere with ion channels. "
        "Relevant to: hERG, ClinTox."
    ),
    "Developmental and mitochondrial toxicity": (
        "Developmental / mitochondrial toxicity",
        "Disrupts mitochondrial function or embryonic development. "
        "Relevant to: ClinTox."
    ),
}

# Properties to skip (not relevant to our TDC tasks or redundant with RDKit PAINS)
_TOXALERTS_SKIP = {
    "PAINS compounds", "Extended Functional Groups (EFG)", "Custom filters",
    "AlphaScreen-GST-FHs", "AlphaScreen-HIS-FHs", "AlphaScreen-FHs",
    "Biodegradable compounds", "Nonbiodegradable compounds",
    "Acute Aquatic Toxicity", "Promiscuity",
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


@lru_cache(maxsize=1)
def _load_toxalerts():
    """Load and compile ToxAlerts SMARTS patterns from CSV.

    Returns list of (alert_id, name, property, compiled_pattern) tuples,
    or empty list if CSV not found.
    """
    from rdkit import Chem

    csv_path = os.path.join(_CACHE_DIR, "toxalerts.csv")
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r") as f:
        lines = [l for l in f if l.strip()]

    reader = csv.DictReader(lines)
    alerts = []
    for r in reader:
        smarts = (r.get("SMARTS") or "").strip()
        prop = (r.get("PROPERTY") or "").strip()
        if not smarts or smarts == "[ERROR]" or prop in _TOXALERTS_SKIP:
            continue
        if prop not in _TOXALERTS_ENDPOINT_MAP:
            continue
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            alerts.append((
                (r.get("Alert ID") or "").strip(),
                (r.get("NAME") or "").strip(),
                prop,
                pat,
            ))
    return alerts


def _screen_toxalerts(smiles: str) -> Optional[str]:
    """Screen against ToxAlerts endpoint-specific SMARTS patterns."""
    from rdkit import Chem
    from collections import OrderedDict

    alerts = _load_toxalerts()
    if not alerts:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Find matching alerts, grouped by endpoint
    grouped: OrderedDict[str, list] = OrderedDict()
    for alert_id, name, prop, pat in alerts:
        if mol.HasSubstructMatch(pat):
            if prop not in grouped:
                grouped[prop] = []
            grouped[prop].append(name)

    if not grouped:
        return "Endpoint-Specific Alerts (ToxAlerts):\n- No endpoint-specific alerts found"

    total_hits = sum(len(v) for v in grouped.values())
    lines = [f"Endpoint-Specific Alerts (ToxAlerts, {total_hits} hits across {len(grouped)} endpoints):"]

    for prop, alert_names in grouped.items():
        display_name, note = _TOXALERTS_ENDPOINT_MAP[prop]
        # Deduplicate and show unique alert names
        unique_names = list(dict.fromkeys(alert_names))
        lines.append(f"- {display_name} ({len(unique_names)} alerts)")
        lines.append(f"  Why: {note}")
        # Show up to 5 alert names
        shown = unique_names[:5]
        lines.append(f"  Alerts: {', '.join(shown)}")
        if len(unique_names) > 5:
            lines.append(f"  ... and {len(unique_names) - 5} more")

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────

def screen_toxicophores(smiles: str) -> str:
    """
    Screen a molecule for toxicophores (structural alerts grouped by
    mechanism of toxicity) and pharmacophore features.

    Alerts from multiple filter libraries (PAINS, Brenk, NIH, ZINC, ChEMBL)
    are deduplicated and grouped into mechanistic categories such as
    'Michael acceptor', 'alkylating agent', or 'PAH', each with an
    explanation of why the pattern is toxicologically relevant.

    Additionally screens against ToxAlerts (OCHEM) endpoint-specific
    SMARTS patterns to identify alerts linked to specific toxicity
    endpoints (skin sensitization, mutagenicity, DILI, etc.).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with toxicophore analysis.
    """
    sections = []

    try:
        sections.append(_screen_structural_alerts(smiles))
    except Exception as e:
        sections.append(f"Toxicophore Screening: Error - {e}")

    try:
        ta_result = _screen_toxalerts(smiles)
        if ta_result:
            sections.append(f"\n{ta_result}")
    except Exception as e:
        sections.append(f"\nEndpoint-Specific Alerts: Error - {e}")

    try:
        sections.append(f"\n{_get_pharmacophore_counts(smiles)}")
    except Exception as e:
        sections.append(f"\nPharmacophore Features: Error - {e}")

    return "\n".join(sections)


def _screen_structural_alerts(smiles: str) -> str:
    """Screen against all RDKit alert catalogs, grouped by mechanism."""
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from collections import OrderedDict

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
    fc = FilterCatalog(params)

    # Collect all raw alerts (deduplicated)
    seen = set()
    raw_alerts = []
    for entry in fc.GetMatches(mol):
        desc = entry.GetDescription().strip()
        desc_norm = desc.lower().replace("_", " ")
        if desc_norm not in seen:
            seen.add(desc_norm)
            raw_alerts.append(desc)

    if not raw_alerts:
        return "Toxicophore Screening:\n- No structural alerts found"

    # Group into semantic categories
    grouped: OrderedDict[str, list] = OrderedDict()
    category_notes: dict = {}
    uncategorized = []

    for alert in raw_alerts:
        cat = _classify_alert(alert)
        if cat:
            if cat not in grouped:
                grouped[cat] = []
                # Find the note for this category
                for name, note, _ in _TOXICOPHORE_CATEGORIES:
                    if name == cat:
                        category_notes[cat] = note
                        break
            grouped[cat].append(alert)
        else:
            uncategorized.append(alert)

    lines = [f"Toxicophore Screening ({len(raw_alerts)} raw alerts → {len(grouped)} categories):"]

    for cat_name, alerts in grouped.items():
        note = category_notes.get(cat_name, "")
        lines.append(f"- {cat_name} ({len(alerts)} alerts)")
        lines.append(f"  Why: {note}")

    if uncategorized:
        lines.append(f"- Other ({len(uncategorized)} alerts): {', '.join(uncategorized[:5])}")
        if len(uncategorized) > 5:
            lines.append(f"  ... and {len(uncategorized) - 5} more")

    return "\n".join(lines)


def _get_pharmacophore_counts(smiles: str) -> str:
    """Extract pharmacophore feature counts."""
    import os
    from collections import Counter
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    features = factory.GetFeaturesForMol(mol)

    counts = Counter()
    for feat in features:
        counts[feat.GetFamily()] += 1

    lines = ["Pharmacophore Feature Counts:"]
    if counts:
        for family, count in counts.items():
            lines.append(f"- {family}: {count}")
    else:
        lines.append("- None detected")

    return "\n".join(lines)


# Keep backward compatibility
screen_safety = screen_toxicophores


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "screen_toxicophores",
        "description": (
            "Screen a molecule for toxicophores — structural motifs associated with "
            "specific toxicity mechanisms. Groups alerts into categories like 'Michael "
            "acceptor' (protein-reactive electrophile → skin sensitization), 'alkylating "
            "agent' (DNA-reactive → mutagenicity), 'PAH' (CYP-activated → carcinogenicity), "
            "etc. Each category includes a mechanistic explanation. Also screens against "
            "ToxAlerts endpoint-specific patterns to flag alerts linked to skin sensitization, "
            "mutagenicity (AMES), carcinogenicity, DILI (reactive metabolites), hERG "
            "(chelators), and ClinTox. Returns pharmacophore feature counts "
            "(donor/acceptor/aromatic/hydrophobe). Use this to identify structural "
            "liabilities and connect them to specific toxicity endpoints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."}
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
