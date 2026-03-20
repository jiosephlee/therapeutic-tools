"""
Tool 4: ADME Properties — ionization, partitioning, solubility.

These properties are causally linked: pKa → ionization → logD → permeability.
TPSA and surface area live in get_molecule_profile / get_3d_properties.
"""

from typing import Dict, Any


def assess_adme_properties(smiles: str, ph: float = 7.4) -> str:
    """
    Assess ADME-related properties: ionization, partitioning, solubility, and surface area.

    These properties are causally linked: pKa → ionization → logD → permeability.

    Args:
        smiles: SMILES string of the molecule.
        ph: Target pH for ionization/logD estimation (default: 7.4).

    Returns:
        Multi-line formatted string with ADME property assessment.
    """
    from .legacy_tools.pka_related_tools import predict_pka, estimate_logd
    from .legacy_tools.RDKit_tools import classify_ionization, get_esol
    from . import metadata_cache

    cached = metadata_cache.lookup_row(smiles)
    sections = []

    # pKa prediction — use cache if available
    if cached and "most_acidic_pka" in cached and "num_acidic_sites" in cached:
        lines = ["pKa Prediction:"]
        lines.append(f"  Most acidic pKa: {cached.get('most_acidic_pka', 'N/A'):.4f}")
        if "most_basic_pka" in cached:
            lines.append(f"  Most basic pKa: {cached['most_basic_pka']:.4f}")
        lines.append(f"  Acidic sites: {int(cached.get('num_acidic_sites', 0))}")
        lines.append(f"  Basic sites: {int(cached.get('num_basic_sites', 0))}")
        sections.append("\n".join(lines))
    else:
        try:
            pka_result = predict_pka(smiles)
            sections.append("pKa Prediction:\n" + pka_result)
        except Exception as e:
            sections.append(f"pKa Prediction: Error - {e}")

    # Ionization classification
    try:
        ionization_result = classify_ionization(smiles, ph=ph)
        sections.append(ionization_result)
    except Exception as e:
        sections.append(f"\nIonization Classification: Error - {e}")

    # LogD estimation — use cache if at default pH
    if cached and "logD_74" in cached and abs(ph - 7.4) < 0.01:
        sections.append(f"\nLogD Estimation:\n  logD at pH 7.4: {cached['logD_74']:.4f}")
    else:
        try:
            logd_result = estimate_logd(smiles, ph=ph)
            sections.append("\nLogD Estimation:\n" + logd_result)
        except Exception as e:
            sections.append(f"\nLogD Estimation: Error - {e}")

    # Solubility — use ML prediction from cache if available, else ESOL heuristic
    if cached and "minimol_solubility_log_mol_L" in cached:
        sections.append(f"\nSolubility (ML prediction): logS = {cached['minimol_solubility_log_mol_L']:.4f} log(mol/L)")
    else:
        try:
            esol_result = get_esol(smiles)
            sections.append(f"\nSolubility (ESOL heuristic): {esol_result}")
        except Exception as e:
            sections.append(f"\nSolubility: Error - {e}")

    return "\n".join(sections)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "assess_adme_properties",
        "description": (
            "Assess ADME properties: pKa prediction (acidic/basic sites), "
            "ionization state classification at target pH, logD estimation, "
            "and aqueous solubility. "
            "These are causally linked: pKa → ionization → logD → permeability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."},
                "ph": {"type": "number", "description": "Target pH for ionization/logD estimation (default: 7.4)."}
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
