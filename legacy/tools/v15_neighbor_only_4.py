"""Version 15 therapeutic tools with only 4-neighbor comparison.

This variant keeps the TRIM local-analog comparison tool, omits the
single-molecule property tool, and requests 2 positive plus 2 negative
neighbors from the TRIM runtime.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

from ..utils.trim import trim_compare_similar_mols


def _compare_tool_description() -> str:
    return (
        "Retrieve text-form local analog evidence. The input SMILES must belong to the active task dataset so the tool can identify "
        "the query split and retrieve the nearest training-set neighbors. The tool returns plain text "
        "with a short definition of query/neighbor/delta, then positive and negative neighbors "
        "(2 positive and 2 negative), each with neighbor/query/delta values for the 36 dense "
        "properties plus functional-group differences."
    )


OPENAI_AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "compare_similar_mols",
            "description": _compare_tool_description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string for the molecule to analyze.",
                    }
                },
                "required": ["smiles"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
COMPARE_SIMILAR_MOLS_TOOL: Dict[str, object] = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])


def compare_similar_mols(smiles: str, task: Optional[str] = None) -> str:
    return trim_compare_similar_mols(
        smiles,
        task=task,
        neighbors_per_label=2,
    )


__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "compare_similar_mols",
]
