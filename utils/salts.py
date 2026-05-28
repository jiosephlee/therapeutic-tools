"""
Tool 8: Salt Removal — preprocessing step.

Strips counterions/salts, returns largest organic fragment.
"""

from typing import Dict, Any


def remove_salts(smiles: str) -> str:
    """
    Remove salts/counterions from a SMILES string.

    Keeps the largest organic fragment and returns its canonical SMILES.
    Use this as a preprocessing step before running other analyses.

    Args:
        smiles: SMILES string (may contain salts, e.g., "CC(=O)O.[Na]").

    Returns:
        Canonical SMILES of the desalted molecule.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"Error: Invalid SMILES: {smiles!r}"
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return ""
    largest = max(fragments, key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()))
    return Chem.MolToSmiles(largest, canonical=True)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remove_salts",
        "description": "Strip salts and counterions, keeping the largest organic fragment. Use when SMILES contains '.' separators.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"],
        }
    }
}
