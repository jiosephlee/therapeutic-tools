"""TRIM runtime adapter endpoints.

Therapeutic-tools does not own TRIM's EBM models or task assets. This module
is the one place that resolves the external TRIM checkout and forwards calls to
TRIM's agent-tool runtime.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Mapping, Optional


DEFAULT_TRIM_ROOT = "/vast/projects/myatskar/design-documents/joseph/trim"


def trim_root() -> str:
    """Return the configured TRIM checkout path."""
    return os.environ.get("TRIM_ROOT", DEFAULT_TRIM_ROOT)


@lru_cache(maxsize=1)
def trim_runtime():
    """Load TRIM's OpenAI-style agent-tool runtime singleton."""
    root = trim_root()
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


def _require_task(task: Optional[str]) -> str:
    if not task:
        raise ValueError(
            "TRIM tool calls require a task such as 'AMES' or 'BBB_Martins'. "
            "The LLM-facing executor should inject this via runtime state."
        )
    return task


def trim_call_tool(
    tool_name: str,
    arguments: str | Mapping[str, Any],
    *,
    task: Optional[str],
    neighbors_per_label: int = 3,
) -> str:
    """Call a TRIM agent tool with explicit task runtime context."""
    return trim_runtime().call_tool(
        tool_name,
        arguments,
        task=_require_task(task),
        neighbors_per_label=neighbors_per_label,
    )


def trim_get_mol_properties_and_fg(smiles: str, *, task: Optional[str]) -> str:
    """Return TRIM's single-molecule property/functional-group text."""
    return trim_call_tool(
        "get_mol_properties_and_fg",
        {"smiles": smiles},
        task=task,
    )


def trim_compare_similar_mols(
    smiles: str,
    *,
    task: Optional[str],
    neighbors_per_label: int = 3,
) -> str:
    """Return TRIM's task-bound local-neighbor comparison text."""
    return trim_call_tool(
        "compare_similar_mols",
        {"smiles": smiles},
        task=task,
        neighbors_per_label=neighbors_per_label,
    )


__all__ = [
    "DEFAULT_TRIM_ROOT",
    "trim_call_tool",
    "trim_compare_similar_mols",
    "trim_get_mol_properties_and_fg",
    "trim_root",
    "trim_runtime",
]
