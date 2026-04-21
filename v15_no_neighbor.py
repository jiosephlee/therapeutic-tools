"""Version 15 therapeutic tools without neighbor lookup.

This variant keeps the TRIM single-molecule evidence tool and omits the
neighbor-comparison tool from the exposed schema surface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

from .v15 import (
    GET_MOL_PROPERTIES_AND_FG_TOOL as _V15_GET_MOL_PROPERTIES_AND_FG_TOOL,
    get_mol_properties_and_fg,
)

OPENAI_AGENT_TOOL_SCHEMAS = [deepcopy(_V15_GET_MOL_PROPERTIES_AND_FG_TOOL)]
GET_MOL_PROPERTIES_AND_FG_TOOL: Dict[str, object] = deepcopy(OPENAI_AGENT_TOOL_SCHEMAS[0])

__all__ = [
    "GET_MOL_PROPERTIES_AND_FG_TOOL",
    "OPENAI_AGENT_TOOL_SCHEMAS",
    "get_mol_properties_and_fg",
]
