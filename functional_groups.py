"""
Tool 2: Functional Group Analysis — AccFG decomposition.

Returns named functional groups with fragment SMILES, attachment points,
and atom IDs for structural reasoning.
"""

from typing import Dict, Any


def analyze_functional_groups(smiles: str) -> str:
    """
    Analyze functional groups in a molecule using AccFG.

    Returns named functional groups with fragment SMILES, attachment points,
    and atom IDs for structural reasoning.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with functional group analysis.
    """
    from .legacy_tools.AccFG import cached_concise_fg_description
    return cached_concise_fg_description(smiles)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_functional_groups",
        "description": (
            "Analyze functional groups in a molecule using AccFG. Returns named functional "
            "groups with fragment SMILES, attachment points, and atom IDs for structural "
            "reasoning. Use this to understand what chemical motifs are present."
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
