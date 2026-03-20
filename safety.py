"""
Tool 6: Safety Screening — structural alerts and pharmacophore features.
"""

from typing import Dict, Any


def screen_safety(smiles: str) -> str:
    """
    Screen a molecule for structural alerts and pharmacophore features.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with safety screening results.
    """
    sections = []

    # Structural alerts — deduplicated across all libraries
    try:
        sections.append(_screen_structural_alerts(smiles))
    except Exception as e:
        sections.append(f"Structural Alerts: Error - {e}")

    # Pharmacophore feature counts (no per-atom details)
    try:
        sections.append(f"\n{_get_pharmacophore_counts(smiles)}")
    except Exception as e:
        sections.append(f"\nPharmacophore Features: Error - {e}")

    return "\n".join(sections)


def _screen_structural_alerts(smiles: str) -> str:
    """Screen against all RDKit alert catalogs, deduplicated."""
    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
    fc = FilterCatalog(params)

    # Deduplicate alerts by normalized description
    seen = set()
    unique_alerts = []
    for entry in fc.GetMatches(mol):
        desc = entry.GetDescription().strip().lower().replace("_", " ")
        if desc not in seen:
            seen.add(desc)
            unique_alerts.append(entry.GetDescription())

    lines = ["Structural Alert Screening:"]
    if unique_alerts:
        lines.append(f"- {len(unique_alerts)} unique alert(s) matched:")
        for alert in unique_alerts:
            lines.append(f"  - {alert}")
    else:
        lines.append("- No structural alerts found")

    return "\n".join(lines)


def _get_pharmacophore_counts(smiles: str) -> str:
    """Extract pharmacophore feature counts (no per-atom details)."""
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


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "screen_safety",
        "description": (
            "Screen a molecule for safety concerns: structural alerts (PAINS, Brenk, NIH, "
            "ZINC, ChEMBL filters) and pharmacophore features (donor/acceptor/aromatic/hydrophobe "
            "with atom IDs). Use get_electronic_properties for charge/EState analysis."
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
