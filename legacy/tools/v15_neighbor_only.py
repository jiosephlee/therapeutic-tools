"""Version 15 therapeutic tools with only neighbor comparison.

This variant keeps the TRIM local-analog comparison tool and omits the
single-molecule property tool from the exposed schema surface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

from .v15 import (
    COMPARE_SIMILAR_MOLS_TOOL as _V15_COMPARE_SIMILAR_MOLS_TOOL,
    compare_similar_mols,
)

OPENAI_AGENT_TOOL_SCHEMAS = [deepcopy(_V15_COMPARE_SIMILAR_MOLS_TOOL)]
COMPARE_SIMILAR_MOLS_TOOL: Dict[str, object] = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])

__all__ = [
    "COMPARE_SIMILAR_MOLS_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "compare_similar_mols",
]
