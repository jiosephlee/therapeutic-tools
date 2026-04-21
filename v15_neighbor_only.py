"""Version 15 therapeutic tools with only neighbor comparison.

This variant keeps the TRIM local-analog comparison tool and omits the
single-molecule property tool from the exposed schema surface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

from .v15 import (
    COMPARE_SIMILAR_MOLS_TOOL as _V15_COMPARE_SIMILAR_MOLS_TOOL,
    TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS as _V15_TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS,
    compare_similar_mols,
)

OPENAI_AGENT_TOOL_SCHEMAS = [deepcopy(_V15_COMPARE_SIMILAR_MOLS_TOOL)]
COMPARE_SIMILAR_MOLS_TOOL: Dict[str, object] = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])
TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS: Dict[str, Dict[str, object]] = {
    task: deepcopy(schema)
    for task, schema in _V15_TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS.items()
}

__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS",
    "compare_similar_mols",
]
