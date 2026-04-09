"""
Tool 2: Functional Group Analysis — AccFG decomposition.

Returns named functional groups with fragment SMILES, attachment points,
and atom IDs for structural reasoning.
"""

from typing import Dict, Any


def analyze_functional_groups(smiles: str, simple: bool = False) -> str:
    """
    Analyze functional groups in a molecule using AccFG.

    Returns named functional groups with fragment SMILES, attachment points,
    and atom IDs for structural reasoning. In ``simple`` mode, returns only
    the functional-group names.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with functional group analysis.
    """
    from .legacy_tools.AccFG import cached_concise_fg_description

    desc = cached_concise_fg_description(smiles)
    if not simple:
        return desc

    names = []
    for line in desc.splitlines():
        line = line.strip()
        if line.startswith("- "):
            names.append(line[2:].split(":")[0].strip())

    if not names:
        return "Functional groups:\n- None detected."
    return "Functional groups:\n- " + "\n- ".join(names)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_functional_groups",
        "description": "Identify functional groups and chemical motifs with fragment SMILES and attachment points.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"],
        }
    }
}
