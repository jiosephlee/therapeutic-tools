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
    from .legacy_tools.RDKit_tools import classify_ionization, get_esol
    from . import metadata_cache

    cached = metadata_cache.lookup_row(smiles)
    sections = []

    # pKa prediction — use cache if available, else compute
    pka_data = _get_pka_data(smiles, cached)
    lines = ["pKa Prediction:"]
    if pka_data["most_acidic_pka"] is not None:
        lines.append(f"  Most acidic pKa: {pka_data['most_acidic_pka']:.2f}")
    if pka_data["most_basic_pka"] is not None:
        lines.append(f"  Most basic pKa: {pka_data['most_basic_pka']:.2f}")
    lines.append(f"  Acidic sites: {pka_data['num_acidic_sites']}, Basic sites: {pka_data['num_basic_sites']}")
    if not pka_data["most_acidic_pka"] and not pka_data["most_basic_pka"]:
        lines.append("  No ionizable sites predicted.")
    sections.append("\n".join(lines))

    # Ionization classification
    try:
        ionization_result = classify_ionization(smiles, ph=ph)
        sections.append(ionization_result)
    except Exception as e:
        sections.append(f"\nIonization Classification: Error - {e}")

    # LogD estimation — computed from pKa data, no redundant recomputation
    logd = _estimate_logd_from_pka(smiles, ph, pka_data, cached)
    sections.append(f"\nLogD at pH {ph}: {logd:.2f}")

    # Solubility — use ML prediction from cache if available, else ESOL heuristic
    if cached and "minimol_solubility_log_mol_L" in cached:
        sections.append(f"Solubility: logS = {cached['minimol_solubility_log_mol_L']:.2f} log(mol/L)")
    else:
        try:
            esol_result = get_esol(smiles)
            sections.append(f"Solubility (ESOL heuristic): {esol_result}")
        except Exception as e:
            sections.append(f"Solubility: Error - {e}")

    return "\n".join(sections)


def _get_pka_data(smiles: str, cached) -> dict:
    """Extract pKa data from cache or compute via MolGpKa. Returns dict with keys:
    most_acidic_pka, most_basic_pka, num_acidic_sites, num_basic_sites."""
    if cached and "most_acidic_pka" in cached and "num_acidic_sites" in cached:
        return {
            "most_acidic_pka": cached.get("most_acidic_pka"),
            "most_basic_pka": cached.get("most_basic_pka"),
            "num_acidic_sites": int(cached.get("num_acidic_sites", 0)),
            "num_basic_sites": int(cached.get("num_basic_sites", 0)),
        }
    # Compute live
    from .legacy_tools.pka_related_tools import _mol_from_smiles, _get_pka_predictor
    mol = _mol_from_smiles(smiles)
    predictor = _get_pka_predictor()
    prediction = predictor.predict(mol)
    base_sites = prediction.base_sites_1
    acid_sites = prediction.acid_sites_1
    return {
        "most_acidic_pka": min(acid_sites.values()) if acid_sites else None,
        "most_basic_pka": max(base_sites.values()) if base_sites else None,
        "num_acidic_sites": len(acid_sites),
        "num_basic_sites": len(base_sites),
    }


def _estimate_logd_from_pka(smiles: str, ph: float, pka_data: dict, cached) -> float:
    """Estimate logD using Henderson-Hasselbalch from pKa data already computed."""
    import math
    from rdkit.Chem import Crippen
    from rdkit import Chem

    if cached and "logD_74" in cached and abs(ph - 7.4) < 0.01:
        return float(cached["logD_74"])

    mol = Chem.MolFromSmiles(smiles)
    logp = float(Crippen.MolLogP(mol))

    f_neutral = 1.0
    if pka_data["most_basic_pka"] is not None:
        f_neutral *= 1.0 / (1.0 + 10.0 ** (pka_data["most_basic_pka"] - ph))
    if pka_data["most_acidic_pka"] is not None:
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - pka_data["most_acidic_pka"]))

    f_neutral = min(1.0, max(1e-12, f_neutral))
    return logp + math.log10(f_neutral)


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
