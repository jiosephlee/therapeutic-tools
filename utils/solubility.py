"""
Tool: Aqueous Solubility Prediction (LogS).

Uses MiniMol embeddings + Ridge regression when available,
falls back to the Delaney ESOL heuristic (RDKit-only).

The MiniMol model is trained on AqSolDB (~9.9k molecules) and cached at
  tools/therapeutic_tools/cache/minimol_solubility/minimol_solubility_ridge.pkl
"""

import math
from typing import Dict, Any

_minimol_predictor = None
_minimol_import_failed = False


def _esol(smiles: str) -> float:
    """
    Delaney ESOL heuristic: log S (mol/L) from RDKit descriptors.
    log S = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAromaticAtoms())
    ap = aromatic / heavy if heavy else 0.0
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def _get_minimol_predictor():
    """Lazy-load the MiniMol solubility predictor (singleton)."""
    global _minimol_predictor, _minimol_import_failed
    if _minimol_import_failed:
        return None
    if _minimol_predictor is not None:
        return _minimol_predictor
    try:
        from .minimol_solubility import MiniMolSolubility

        _minimol_predictor = MiniMolSolubility()
        return _minimol_predictor
    except Exception:
        _minimol_import_failed = True
        return None


def _classify_solubility(log_s: float) -> str:
    """Classify aqueous solubility into pharmacological categories."""
    if log_s >= -1:
        return "highly soluble"
    elif log_s >= -2:
        return "soluble"
    elif log_s >= -3:
        return "moderately soluble"
    elif log_s >= -4:
        return "slightly soluble"
    elif log_s >= -5:
        return "poorly soluble"
    else:
        return "practically insoluble"


def predict_solubility(smiles: str) -> str:
    """
    Predict aqueous solubility (log S in log mol/L) for a molecule.

    Uses MiniMol neural embeddings + Ridge regression trained on AqSolDB
    when available, otherwise falls back to the Delaney ESOL heuristic.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Formatted string with predicted log S value, classification, and method used.
    """
    method = "ESOL heuristic"
    log_s = None
    from . import endpoints

    # Try MiniMol first
    predictor = _get_minimol_predictor()
    if predictor is not None:
        try:
            val = predictor.predict(smiles)
            if not math.isnan(val):
                log_s = val
                method = "MiniMol + Ridge (AqSolDB)"
        except Exception:
            pass

    # Fallback to ESOL
    if log_s is None:
        log_s = endpoints.solubility_log_s(smiles)
        if log_s is None:
            return f"Error: Could not compute solubility for SMILES: {smiles!r}"

    category = _classify_solubility(log_s)

    lines = [
        "Aqueous Solubility Prediction:",
        f"- Predicted log S: {log_s:.2f} log mol/L",
        f"- Classification: {category}",
        f"- Method: {method}",
    ]

    # Add context for drug-likeness interpretation
    if log_s < -5:
        lines.append("- Note: Very low solubility may limit oral bioavailability and require formulation strategies.")
    elif log_s < -3:
        lines.append("- Note: Moderate-to-low solubility; may need salt forms or enabling formulations for adequate exposure.")

    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "predict_solubility",
        "description": (
            "Predict aqueous solubility (log S in log mol/L) for a molecule. "
            "Returns the predicted value, a pharmacological classification "
            "(highly soluble to practically insoluble), and notes on bioavailability "
            "implications. Solubility is a key determinant of oral drug absorption."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                }
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}
