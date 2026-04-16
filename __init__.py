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

from .safety import screen_structural_alerts, screen_toxicophores, screen_safety  # backward-compat aliases
from .safety import TOOL_SCHEMA as SCREEN_STRUCTURAL_ALERTS_TOOL

from .similarity import find_similar_molecules
from .similarity import TOOL_SCHEMA as FIND_SIMILAR_MOLECULES_TOOL
from .similarity import TASK_TOOL_SCHEMAS as SIMILAR_MOLECULES_TASK_SCHEMAS
from .similarity import TASK_CALLABLES as SIMILAR_MOLECULES_TASK_CALLABLES

from .salts import remove_salts
from .salts import TOOL_SCHEMA as REMOVE_SALTS_TOOL

from .calculator import evaluate_arithmetic
from .calculator import TOOL_SCHEMA as EVALUATE_ARITHMETIC_TOOL

from .electronic import get_electronic_properties
from .electronic import TOOL_SCHEMA as GET_ELECTRONIC_PROPERTIES_TOOL

from .metabolism import predict_metabolites
from .metabolism import TOOL_SCHEMA as PREDICT_METABOLITES_TOOL

from .scaffold import get_scaffold
from .scaffold import TOOL_SCHEMA as GET_SCAFFOLD_TOOL

from .solubility import predict_solubility
from .solubility import TOOL_SCHEMA as PREDICT_SOLUBILITY_TOOL

from .decision_tree import decision_tree_analysis
from .decision_tree import TOOL_SCHEMA as DECISION_TREE_ANALYSIS_TOOL

from .v10 import get_molecular_properties, get_similar_neighbors
from .v10 import (
    GET_MOLECULAR_PROPERTIES_TOOL,
    GET_SIMILAR_NEIGHBORS_TOOL,
    TASK_NEIGHBOR_TOOL_SCHEMAS as V10_TASK_NEIGHBOR_TOOL_SCHEMAS,
    TASK_NEIGHBOR_CALLABLES as V10_TASK_NEIGHBOR_CALLABLES,
)

from .v11 import get_features, get_neighbors
from .v11 import (
    GET_FEATURES_TOOL,
    GET_NEIGHBORS_TOOL,
    FEATURE_NAMES as V11_FEATURE_NAMES,
    TASK_NEIGHBOR_TOOL_SCHEMAS as V11_TASK_NEIGHBOR_TOOL_SCHEMAS,
    TASK_NEIGHBOR_CALLABLES as V11_TASK_NEIGHBOR_CALLABLES,
)
from .v12 import get_features as v12_get_features, get_neighbors as v12_get_neighbors
from .v12 import (
    GET_FEATURES_TOOL as V12_GET_FEATURES_TOOL,
    GET_NEIGHBORS_TOOL as V12_GET_NEIGHBORS_TOOL,
    FEATURE_NAMES as V12_FEATURE_NAMES,
    TASK_NEIGHBOR_TOOL_SCHEMAS as V12_TASK_NEIGHBOR_TOOL_SCHEMAS,
    TASK_NEIGHBOR_CALLABLES as V12_TASK_NEIGHBOR_CALLABLES,
)
from .v13 import get_features as v13_get_features, get_neighbors as v13_get_neighbors
from .v13 import (
    GET_FEATURES_TOOL as V13_GET_FEATURES_TOOL,
    GET_NEIGHBORS_TOOL as V13_GET_NEIGHBORS_TOOL,
    FEATURE_NAMES as V13_FEATURE_NAMES,
    TOP20_RDKIT_DESCRIPTORS as V13_TOP20_RDKIT_DESCRIPTORS,
    TASK_NEIGHBOR_TOOL_SCHEMAS as V13_TASK_NEIGHBOR_TOOL_SCHEMAS,
    TASK_NEIGHBOR_CALLABLES as V13_TASK_NEIGHBOR_CALLABLES,
)
from .v14 import get_features as v14_get_features, get_neighbors as v14_get_neighbors
from .v14 import (
    GET_FEATURES_TOOL as V14_GET_FEATURES_TOOL,
    GET_NEIGHBORS_TOOL as V14_GET_NEIGHBORS_TOOL,
    FEATURE_NAMES as V14_FEATURE_NAMES,
    V14_ADDITIONAL_FEATURES,
    TASK_NEIGHBOR_TOOL_SCHEMAS as V14_TASK_NEIGHBOR_TOOL_SCHEMAS,
    TASK_NEIGHBOR_CALLABLES as V14_TASK_NEIGHBOR_CALLABLES,
)
from .v14_no_neighbor import get_features as v14_no_neighbor_get_features
from .v14_no_neighbor import (
    GET_FEATURES_TOOL as V14_NO_NEIGHBOR_GET_FEATURES_TOOL,
    FEATURE_NAMES as V14_NO_NEIGHBOR_FEATURE_NAMES,
)
from .v15 import get_mol_properties_and_fg, compare_similar_mols
from .v15 import (
    GET_MOL_PROPERTIES_AND_FG_TOOL as V15_GET_MOL_PROPERTIES_AND_FG_TOOL,
    COMPARE_SIMILAR_MOLS_TOOL as V15_COMPARE_SIMILAR_MOLS_TOOL,
    TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS as V15_TASK_COMPARE_SIMILAR_MOLS_TOOL_SCHEMAS,
)


# ============================================================
# Tool registry (default tools served to agents)
# ============================================================

CONSOLIDATED_TOOLS: List[Dict[str, Any]] = [
    GET_MOLECULE_PROFILE_TOOL,
    ANALYZE_FUNCTIONAL_GROUPS_TOOL,
    ANALYZE_RING_SYSTEMS_TOOL,
    ASSESS_ADME_PROPERTIES_TOOL,
    SCREEN_STRUCTURAL_ALERTS_TOOL,
    FIND_SIMILAR_MOLECULES_TOOL,
    GET_MOLECULAR_PROPERTIES_TOOL,
    *V10_TASK_NEIGHBOR_TOOL_SCHEMAS.values(),
    REMOVE_SALTS_TOOL,
    PREDICT_METABOLITES_TOOL,
]

_FUNCTION_MAP = {
    "get_molecule_profile": get_molecule_profile,
    "analyze_functional_groups": analyze_functional_groups,
    "analyze_ring_systems": analyze_ring_systems,
    "assess_adme_properties": assess_adme_properties,
    "get_3d_properties": get_3d_properties,
    "screen_structural_alerts": screen_structural_alerts,
    "find_similar_molecules": find_similar_molecules,
    "remove_salts": remove_salts,
    "evaluate_arithmetic": evaluate_arithmetic,
    "get_electronic_properties": get_electronic_properties,
    "predict_metabolites": predict_metabolites,
    "get_scaffold": get_scaffold,
    "predict_solubility": predict_solubility,
    "decision_tree_analysis": decision_tree_analysis,
    "get_molecular_properties": get_molecular_properties,
    "get_similar_neighbors": get_similar_neighbors,
    **V10_TASK_NEIGHBOR_CALLABLES,
    **SIMILAR_MOLECULES_TASK_CALLABLES,
    "get_features": get_features,
    "get_neighbors": get_neighbors,
    **V11_TASK_NEIGHBOR_CALLABLES,
    "get_mol_properties_and_fg": get_mol_properties_and_fg,
    "compare_similar_mols": compare_similar_mols,
}


def get_function_by_name(name: str):
    """Look up a tool function by its name."""
    return _FUNCTION_MAP.get(name)
