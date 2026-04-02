"""
Decision Tree Analysis Tool.

Looks up precomputed Random Forest feature attributions from LLM4SD/RF_explanations.
Returns SHAP-style feature importance showing which molecular features drive the
RF classifier's prediction for a given molecule.

The RF models are task-specific — trained per TDC binary classification task with
engineered molecular features (structural alerts, physicochemical descriptors, etc.).
"""

import json
import os
from typing import Any, Dict, Optional

_RF_CACHE: Dict[str, Dict[str, Any]] = {}  # {task: {smiles: entry}}
_RF_DIR: Optional[str] = None


def _get_rf_dir() -> str:
    """Resolve the RF_explanations directory."""
    global _RF_DIR
    if _RF_DIR is not None:
        return _RF_DIR

    # Check environment variable first
    env_dir = os.environ.get("RF_EXPLANATIONS_DIR")
    if env_dir and os.path.isdir(env_dir):
        _RF_DIR = env_dir
        return _RF_DIR

    # Default: LLM4SD/RF_explanations relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    _RF_DIR = os.path.join(project_root, "LLM4SD", "RF_explanations")
    return _RF_DIR


def _load_task_cache(task: str) -> Dict[str, Any]:
    """Lazy-load RF explanations for a task, keyed by SMILES."""
    if task in _RF_CACHE:
        return _RF_CACHE[task]

    from rdkit import Chem

    rf_dir = _get_rf_dir()
    lookup: Dict[str, Any] = {}

    for split_file in ("train.jsonl", "valid.jsonl"):
        path = os.path.join(rf_dir, task, split_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    smi = entry["drug"]
                    lookup[smi] = entry
                    # Also index by canonical SMILES
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        canon = Chem.MolToSmiles(mol)
                        if canon != smi:
                            lookup[canon] = entry
                except Exception:
                    pass

    _RF_CACHE[task] = lookup
    return lookup


# Internal arg `include_pseudo_label` is NOT exposed in the OpenAI tool schema.
# When True, appends the RF's predicted class label (majority vote) to the output.
# Used by prepended-tools build scripts for the "pseudo" dataset variant;
# the model never sees this parameter at runtime.
def decision_tree_analysis(
    smiles: str,
    task: str = "",
    include_pseudo_label: bool = False,
) -> str:
    """
    Look up Random Forest feature attributions for a molecule.

    Args:
        smiles: SMILES string of the molecule.
        task: TDC task name (e.g. "AMES", "hERG"). If empty, searches all cached tasks.
        include_pseudo_label: Internal flag. Append RF predicted label to output.

    Returns:
        Formatted string with RF predicted probabilities and top feature attributions.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"Error: Invalid SMILES: {smiles!r}"
    canon = Chem.MolToSmiles(mol)

    entry = None

    # If task specified, look up directly
    if task:
        cache = _load_task_cache(task)
        entry = cache.get(canon) or cache.get(smiles)
    else:
        # Search all cached tasks
        rf_dir = _get_rf_dir()
        if os.path.isdir(rf_dir):
            for task_name in os.listdir(rf_dir):
                task_path = os.path.join(rf_dir, task_name)
                if not os.path.isdir(task_path):
                    continue
                cache = _load_task_cache(task_name)
                entry = cache.get(canon) or cache.get(smiles)
                if entry is not None:
                    break

    if entry is None:
        return "No decision tree analysis available for this molecule."

    explanation = entry["explanation"]

    # Strip "Predicted class:" line unless pseudo label requested —
    # that line leaks the RF prediction to the model.
    if not include_pseudo_label:
        lines = explanation.split("\n")
        lines = [l for l in lines if not l.startswith("Predicted class:")]
        explanation = "\n".join(lines)
    else:
        pseudo = entry.get("pseudo_label")
        label_str = "(B)" if pseudo == 1 else "(A)"
        explanation += f"\nRF Predicted Label: {label_str}"

    return explanation


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "decision_tree_analysis",
        "description": (
            "Run a trained Random Forest model on the molecule and return the "
            "predicted class probabilities with SHAP-style feature attributions "
            "showing which molecular features drive the prediction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule",
                },
            },
            "required": ["smiles"],
        },
    },
}
