"""
Version 15 therapeutic tools — thin wrapper over TRIM's agent-tool runtime.

Both callables (`get_mol_properties_and_fg` and `compare_similar_mols`) delegate
to `trim.reasoning.agent_tools.build_openai_tool_runtime()` so eval-time tool
outputs byte-match the training traces produced by TRIM.

TRIM is expected to be an external standalone checkout (no longer a submodule).
The location is resolved from the ``TRIM_ROOT`` environment variable, defaulting
to ``/vast/projects/myatskar/design-documents/joseph/trim``.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from functools import lru_cache
from typing import Dict, Optional

from .similarity import TASKS

_DEFAULT_TRIM_ROOT = "/vast/projects/myatskar/design-documents/joseph/trim"


def _trim_root() -> str:
    return os.environ.get("TRIM_ROOT", _DEFAULT_TRIM_ROOT)


@lru_cache(maxsize=1)
def _trim_runtime():
    root = _trim_root()
    src = os.path.join(root, "src")
    if not os.path.isdir(src):
        raise RuntimeError(
            f"TRIM checkout not found at TRIM_ROOT={root} (expected {src}). "
            "Set TRIM_ROOT to the standalone TRIM clone."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    from trim.reasoning.agent_tools import build_openai_tool_runtime

    return build_openai_tool_runtime()


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


def _compare_tool_description(task: str | None) -> str:
    task_scope = f" for task {task}" if task else " for the current task"
    return (
        "Retrieve text-form local analog evidence"
        f"{task_scope}. The input SMILES must belong to that task dataset so the tool can identify "
        "the query split and retrieve the nearest training-set neighbors. The tool returns plain text "
        "with a short definition of query/neighbor/delta, then positive and negative neighbors "
        "(typically 3 positive and 3 negative, depending on the active manifest), each with "
        "neighbor/query/delta values for the 36 dense properties plus functional-group differences."
    )


def build_openai_agent_tool_schemas(*, task: str | None = None) -> list[dict[str, object]]:
    shared_parameters = _smiles_only_parameters()
    return [
        {
            "type": "function",
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
        {
            "type": "function",
            "name": "compare_similar_mols",
            "description": _compare_tool_description(task),
            "parameters": deepcopy(shared_parameters),
            "strict": True,
        },
    ]


OPENAI_AGENT_TOOL_SCHEMAS = build_openai_agent_tool_schemas()
GET_MOL_PROPERTIES_AND_FG_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])
COMPARE_SIMILAR_MOLS_TOOL = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[1])
TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS: Dict[str, Dict[str, object]] = {
    task: build_openai_agent_tool_schemas(task=task)[1] for task in TASKS
}


def get_mol_properties_and_fg(smiles: str, task: Optional[str] = None) -> str:
    return _trim_runtime().call_tool(
        "get_mol_properties_and_fg", {"smiles": smiles}, task=task
    )


def compare_similar_mols(smiles: str, task: Optional[str] = None) -> str:
    if task is None:
        raise ValueError(
            "compare_similar_mols requires a task; the executor must inject "
            "it via extra_state (see MultiTurnAgentExecutor.execute)."
        )
    return _trim_runtime().call_tool(
        "compare_similar_mols", {"smiles": smiles}, task=task
    )


__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "GET_MOL_PROPERTIES_AND_FG_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS",
    "build_openai_agent_tool_schemas",
    "compare_similar_mols",
    "get_mol_properties_and_fg",
]
