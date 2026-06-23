"""
Version 15 therapeutic tools — thin wrapper over TRIM's agent-tool runtime.

Both callables (`get_mol_properties_and_fg` and `compare_similar_mols`) delegate
to `trim.reasoning.agent_tools.build_openai_tool_runtime()` so eval-time tool
outputs byte-match the training traces produced by TRIM.

TRIM is expected to be an external standalone checkout. The location is
resolved from the ``TRIM_ROOT`` environment variable when these legacy tools
are called.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from ..utils.trim import (
    trim_compare_similar_mols,
    trim_get_mol_properties_and_fg,
    trim_runtime as _trim_runtime,
)


def _smiles_only_parameters() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES string for the molecule to analyze.",
            }
        },
        "required": ["smiles"],
        "additionalProperties": False,
    }


def _compare_tool_description() -> str:
    return (
        "Retrieve text-form local analog evidence. The input SMILES must belong to the active task dataset so the tool can identify "
        "the query split and retrieve the nearest training-set neighbors. The tool returns plain text "
        "with a short definition of query/neighbor/delta, then positive and negative neighbors "
        "(typically 3 positive and 3 negative, depending on the active manifest), each with "
        "neighbor/query/delta values for the 36 dense properties plus functional-group differences."
    )


def build_openai_agent_tool_schemas() -> list[dict[str, object]]:
    """Return v15 tool schemas in nested OpenAI form.

    Shape: ``{"type": "function", "function": {"name": ..., "parameters": ...}}``.
    The gpt-oss harmony Jinja chat template indexes ``tool.function.name``; a flat
    shape (``{"type": "function", "name": ...}``) raises
    ``UndefinedError: 'dict object' has no attribute 'function'`` during render.
    """
    shared_parameters = _smiles_only_parameters()
    return [
        {
            "type": "function",
            "function": {
                "name": "get_mol_properties_and_fg",
                "description": (
                    "Return plain-text single-molecule property evidence. The tool outputs one line per "
                    "dense property in 'display_name: value' form for the 36 default dense properties, "
                    "followed by the molecule's present functional groups and their counts. When the "
                    "strongest acidic pKa or strongest basic pKa is undefined, the text explicitly says "
                    "'not applicable (no acidic/basic site)'."
                ),
                "parameters": deepcopy(shared_parameters),
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_similar_mols",
                "description": _compare_tool_description(),
                "parameters": deepcopy(shared_parameters),
                "strict": True,
            },
        },
    ]


OPENAI_AGENT_TOOL_SCHEMAS = build_openai_agent_tool_schemas()
GET_MOL_PROPERTIES_AND_FG_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])
COMPARE_SIMILAR_MOLS_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[1])


def get_mol_properties_and_fg(smiles: str, task: Optional[str] = None) -> str:
    return trim_get_mol_properties_and_fg(smiles, task=task)


def compare_similar_mols(smiles: str, task: Optional[str] = None) -> str:
    return trim_compare_similar_mols(smiles, task=task)


__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "GET_MOL_PROPERTIES_AND_FG_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "build_openai_agent_tool_schemas",
    "compare_similar_mols",
    "get_mol_properties_and_fg",
]
