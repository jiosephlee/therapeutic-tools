"""
therapeutic-tools — Consolidated cheminformatics tools for LLM tool-calling.

Provides semantic tool functions and their OpenAI tool schemas.
Legacy individual descriptor functions are available in .legacy_tools.
"""

from typing import List, Dict, Any

# ============================================================
# Consolidated tools (semantic tool files)
# ============================================================
from .molecule_profile import get_molecule_profile
from .molecule_profile import TOOL_SCHEMA as GET_MOLECULE_PROFILE_TOOL

from .functional_groups import analyze_functional_groups
from .functional_groups import TOOL_SCHEMA as ANALYZE_FUNCTIONAL_GROUPS_TOOL

from .ring_systems import analyze_ring_systems
from .ring_systems import TOOL_SCHEMA as ANALYZE_RING_SYSTEMS_TOOL

from .adme import assess_adme_properties
from .adme import TOOL_SCHEMA as ASSESS_ADME_PROPERTIES_TOOL

from .three_d import get_3d_properties
from .three_d import TOOL_SCHEMA as GET_3D_PROPERTIES_TOOL

from .safety import screen_toxicophores, screen_safety  # screen_safety is backward-compat alias
from .safety import TOOL_SCHEMA as SCREEN_TOXICOPHORES_TOOL

from .similarity import find_similar_molecules
from .similarity import TOOL_SCHEMA as FIND_SIMILAR_MOLECULES_TOOL

from .salts import remove_salts
from .salts import TOOL_SCHEMA as REMOVE_SALTS_TOOL

from .calculator import evaluate_arithmetic
from .calculator import TOOL_SCHEMA as EVALUATE_ARITHMETIC_TOOL

# Expansion tools
from .electronic import get_electronic_properties
from .electronic import TOOL_SCHEMA as GET_ELECTRONIC_PROPERTIES_TOOL

from .metabolism import predict_metabolism_sites
from .metabolism import TOOL_SCHEMA as PREDICT_METABOLISM_SITES_TOOL

from .scaffold import get_scaffold
from .scaffold import TOOL_SCHEMA as GET_SCAFFOLD_TOOL


# ============================================================
# Tool registry
# ============================================================

CONSOLIDATED_TOOLS: List[Dict[str, Any]] = [
    GET_MOLECULE_PROFILE_TOOL,
    ANALYZE_FUNCTIONAL_GROUPS_TOOL,
    ANALYZE_RING_SYSTEMS_TOOL,
    ASSESS_ADME_PROPERTIES_TOOL,
    GET_3D_PROPERTIES_TOOL,
    SCREEN_TOXICOPHORES_TOOL,
    FIND_SIMILAR_MOLECULES_TOOL,
    REMOVE_SALTS_TOOL,
    EVALUATE_ARITHMETIC_TOOL,
    GET_ELECTRONIC_PROPERTIES_TOOL,
    PREDICT_METABOLISM_SITES_TOOL,
    GET_SCAFFOLD_TOOL,
]

_FUNCTION_MAP = {
    "get_molecule_profile": get_molecule_profile,
    "analyze_functional_groups": analyze_functional_groups,
    "analyze_ring_systems": analyze_ring_systems,
    "assess_adme_properties": assess_adme_properties,
    "get_3d_properties": get_3d_properties,
    "screen_toxicophores": screen_toxicophores,
    "find_similar_molecules": find_similar_molecules,
    "remove_salts": remove_salts,
    "evaluate_arithmetic": evaluate_arithmetic,
    "get_electronic_properties": get_electronic_properties,
    "predict_metabolism_sites": predict_metabolism_sites,
    "get_scaffold": get_scaffold,
}


def get_function_by_name(name: str):
    """Look up a tool function by its name."""
    return _FUNCTION_MAP.get(name)
