"""
Tool 7: Similar Molecules — task-aware embedding retrieval + MCS + contrastive example.

Uses precomputed task-specific embeddings stored in
cache/task-specific-embeddings/<task>/embeddings.npz for K-nearest-neighbor
retrieval from the training set.  Embedding files follow the format from
tdc_knn.py:  npz with "smiles", "embeddings", and "labels" arrays.

Also leverages a consolidated metadata.csv (cache/metadata.csv) to surface
precomputed numerical descriptors for each neighbor.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from functools import lru_cache
from sklearn.metrics import accuracy_score, f1_score

import json as _json

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_EMBEDDINGS_DIR = _CACHE_DIR


# ------------------------------------------------------------------
# Embedding & metadata loaders
# ------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_task_data(task: str):
    """Load precomputed embeddings for *task*.

    Returns ``(smiles, embeddings, labels)`` or ``None`` if the file is
    missing.  ``labels`` may itself be ``None`` if the npz was saved
    without them.
    """
    npz_path = os.path.join(_EMBEDDINGS_DIR, task, "embeddings.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    smiles = d["smiles"]
    embeddings = d["embeddings"].astype(np.float32)
    labels = d["labels"] if "labels" in d else None
    return smiles, embeddings, labels


def _load_metadata() -> Optional[pd.DataFrame]:
    """Delegate to shared metadata_cache module."""
    from . import metadata_cache
    return metadata_cache._load_metadata()


@lru_cache(maxsize=1)
def _load_knn_metrics() -> dict:
    """Load precomputed KNN accuracy/F1 per task."""
    path = os.path.join(_CACHE_DIR, "knn_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return _json.load(f)
    return {}


_METADATA_DISPLAY_PROPS = [
    "MolWt", "MolLogP", "TPSA", "qed", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "logD_74",
]


def _get_metadata_line(smiles: str) -> Optional[str]:
    """Return a short summary of cached metadata properties for *smiles*."""
    meta = _load_metadata()
    if meta is None or smiles not in meta.index:
        return None
    row = meta.loc[smiles]
    # If duplicate SMILES, take the first row
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    parts = []
    for prop in _METADATA_DISPLAY_PROPS:
        if prop in row.index and pd.notna(row[prop]):
            val = row[prop]
            parts.append(f"{prop}={val:.2f}" if isinstance(val, float) else f"{prop}={int(val)}")
    return ", ".join(parts) if parts else None


def lookup_metadata_property(smiles: str, prop: str) -> Optional[float]:
    """Look up a single numerical property from the cached metadata.

    Returns the value if the property column exists and the SMILES is
    present, otherwise ``None``.  Other tools can call this before
    recomputing a descriptor from scratch.
    """
    meta = _load_metadata()
    if meta is None or smiles not in meta.index or prop not in meta.columns:
        return None
    val = meta.at[smiles, prop]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val) if pd.notna(val) else None


# ------------------------------------------------------------------
# Core retrieval
# ------------------------------------------------------------------

def find_similar_molecules(smiles: str, task: str, k: int = 5) -> str:
    """
    Find similar molecules using task-aware embeddings.

    Uses precomputed embeddings to retrieve the K nearest neighbors
    from the training set, plus the closest molecule with the opposite label
    (contrastive example). Computes MCS between the query and each neighbor.

    Args:
        smiles: SMILES string of the query molecule.
        task: Task name (e.g., "AMES", "DILI", "BBB_Martins") for task-aware retrieval.
        k: Number of nearest neighbors to retrieve (default: 5).

    Returns:
        Multi-line formatted string with similar molecules, their labels,
        metadata properties, and shared scaffolds (MCS).
    """
    data = _load_task_data(task)
    if data is None:
        return (
            f"Error: No precomputed embeddings found for task '{task}'.\n"
            f"Expected: cache/{task}/embeddings.npz"
        )

    train_smiles, train_embeddings, train_labels = data

    # --- look up query in the precomputed embedding map ---
    match_idx = np.where(train_smiles == smiles)[0]
    if len(match_idx) == 0:
        return (
            f"Error: Query molecule '{smiles}' not found in the "
            f"precomputed embeddings for task '{task}'."
        )
    query_emb = train_embeddings[match_idx[0]]

    # --- cosine similarity ---
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8
    train_normed = train_embeddings / norms
    similarities = train_normed @ query_norm

    # Mask exact SMILES match so it doesn't appear as its own neighbor
    exact_mask = (train_smiles == smiles)
    similarities[exact_mask] = -np.inf

    # Retrieve k=27 neighbors for neighborhood confidence, display top-k
    neighborhood_k = 27
    n_available = int((~exact_mask).sum())
    effective_neighborhood_k = min(neighborhood_k, n_available)
    display_k = min(k, n_available)

    top_neighborhood_idx = np.argsort(similarities)[::-1][:effective_neighborhood_k]
    top_k_idx = top_neighborhood_idx[:display_k]

    neighbors: List[dict] = []
    for idx in top_k_idx:
        neighbors.append({
            "smiles": str(train_smiles[idx]),
            "similarity": float(similarities[idx]),
            "label": int(train_labels[idx]) if train_labels is not None else "N/A",
        })

    # --- Neighborhood confidence: leave-one-out KNN accuracy/F1 on the 27 neighbors ---
    neighborhood_acc = None
    neighborhood_f1 = None
    if train_labels is not None and effective_neighborhood_k >= 3:
        nbr_embs = train_embeddings[top_neighborhood_idx]
        nbr_labels = train_labels[top_neighborhood_idx].astype(int)
        nbr_norms = np.linalg.norm(nbr_embs, axis=1, keepdims=True) + 1e-8
        nbr_normed = nbr_embs / nbr_norms

        # Leave-one-out: for each of the 27 neighbors, predict its label
        # using majority vote of the other 26
        loo_preds = []
        for i in range(effective_neighborhood_k):
            sims_i = nbr_normed @ nbr_normed[i]
            sims_i[i] = -np.inf  # exclude self
            # Use remaining neighbors as voters
            voter_idx = np.argsort(sims_i)[::-1][:effective_neighborhood_k - 1]
            voter_labels = nbr_labels[voter_idx]
            loo_preds.append(int(np.round(voter_labels.mean())))

        neighborhood_acc = accuracy_score(nbr_labels, loo_preds)
        neighborhood_f1 = f1_score(nbr_labels, loo_preds, zero_division=0)

    # --- contrastive example (closest opposite-label molecule) ---
    contrastive = None
    if train_labels is not None:
        # Determine the query's presumed label
        query_idx = np.where(train_smiles == smiles)[0]
        if len(query_idx) > 0:
            query_label = int(train_labels[query_idx[0]])
        elif neighbors:
            # Heuristic: assume same label as nearest neighbor
            query_label = neighbors[0]["label"]
        else:
            query_label = None

        if query_label is not None:
            opposite_mask = (train_labels != query_label) & ~exact_mask
            if np.any(opposite_mask):
                contra_sims = similarities.copy()
                contra_sims[~opposite_mask] = -np.inf
                contra_idx = int(np.argmax(contra_sims))
                contrastive = {
                    "smiles": str(train_smiles[contra_idx]),
                    "similarity": float(similarities[contra_idx]),
                    "label": int(train_labels[contra_idx]),
                }

    return _format_results(
        smiles, task, neighbors, contrastive,
        neighborhood_acc=neighborhood_acc,
        neighborhood_f1=neighborhood_f1,
        neighborhood_k=effective_neighborhood_k,
    )


# ------------------------------------------------------------------
# MCS helper
# ------------------------------------------------------------------

def _compute_mcs_summary(query_smiles: str, neighbor_smiles: str) -> Optional[str]:
    """Compute MCS between query and neighbor. Returns compact summary or None."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS

        mol_q = Chem.MolFromSmiles(query_smiles)
        mol_n = Chem.MolFromSmiles(neighbor_smiles)
        if mol_q is None or mol_n is None:
            return None

        mcs = rdFMCS.FindMCS(
            [mol_q, mol_n],
            timeout=2,
            matchValences=False,
            ringMatchesRingOnly=True,
        )
        if mcs.canceled or mcs.numAtoms == 0:
            return None

        q_atoms = mol_q.GetNumHeavyAtoms()
        n_atoms = mol_n.GetNumHeavyAtoms()
        coverage = mcs.numAtoms / q_atoms if q_atoms > 0 else 0

        return f"{mcs.numAtoms} shared atoms ({coverage:.0%} of query)"
    except Exception:
        return None


# ------------------------------------------------------------------
# Scaffold helper
# ------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Return Murcko scaffold SMILES, cached. Returns None on failure."""
    try:
        from .scaffold import murcko_scaffold_smiles
        s = murcko_scaffold_smiles(smiles)
        return s if s else None
    except Exception:
        return None


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def _get_fg_summary(smiles: str) -> Optional[str]:
    """Get a one-line functional group summary for a neighbor molecule."""
    try:
        from .legacy_tools.AccFG import cached_concise_fg_description
        desc = cached_concise_fg_description(smiles)
        if not desc:
            return None
        # Parse the multi-line FG description into a compact one-liner
        # Lines look like: "- carboxylic acid: C(=O)O ([*]C(=O)O)"
        # or "- carboxylic ester (x2): ..."
        names = []
        for line in desc.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Extract just the name (before the colon)
                name_part = line[2:].split(":")[0].strip()
                names.append(name_part)
        return ", ".join(names) if names else None
    except Exception:
        return None


def _format_neighbor(idx: int, neighbor: dict, mcs_summary: Optional[str]) -> str:
    """Format a single neighbor for output."""
    label_str = neighbor['label']
    line = (
        f"{idx}. {neighbor['smiles']} "
        f"(similarity: {neighbor['similarity']:.4f}, label: {label_str})"
    )
    fg = _get_fg_summary(neighbor['smiles'])
    if fg:
        line += f"\n   Functional groups: {fg}"
    if mcs_summary:
        line += f"\n   MCS: {mcs_summary}"
    # Murcko scaffold
    nbr_scaffold = _get_murcko_scaffold(neighbor['smiles'])
    if nbr_scaffold:
        line += f"\n   Murcko scaffold: {nbr_scaffold}"
    return line


def _format_results(
    query_smiles: str,
    task: str,
    neighbors: List[dict],
    contrastive: Optional[dict],
    neighborhood_acc: Optional[float] = None,
    neighborhood_f1: Optional[float] = None,
    neighborhood_k: int = 27,
) -> str:
    """Format the full similarity search output."""
    # Global KNN model metrics for this task
    knn_metrics = _load_knn_metrics().get(task, {})
    global_acc = knn_metrics.get("accuracy")
    global_f1 = knn_metrics.get("f1")

    header = f"Similar Molecules for task '{task}'"
    if global_acc is not None:
        header += f" (KNN global accuracy={global_acc:.3f}, F1={global_f1:.3f})"
    sections = [header + ":", ""]

    # Query scaffold
    query_scaffold = _get_murcko_scaffold(query_smiles)
    if query_scaffold:
        sections.append(f"Query Murcko scaffold: {query_scaffold}")
        sections.append("")

    # Neighborhood confidence: how well KNN performs locally
    if neighborhood_acc is not None:
        sections.append(
            f"Neighborhood confidence (leave-one-out on {neighborhood_k} nearest neighbors): "
            f"accuracy={neighborhood_acc:.3f}, F1={neighborhood_f1:.3f}"
        )
        sections.append("")

    sections.append("Nearest Neighbors:")
    for i, neighbor in enumerate(neighbors, 1):
        mcs = _compute_mcs_summary(query_smiles, neighbor["smiles"])
        sections.append(_format_neighbor(i, neighbor, mcs))

    if contrastive:
        sections.append("")
        sections.append("Nearest Contrastive Example (opposite label):")
        mcs = _compute_mcs_summary(query_smiles, contrastive["smiles"])
        sections.append(_format_neighbor(len(neighbors) + 1, contrastive, mcs))

    return "\n".join(sections)


# ------------------------------------------------------------------
# Tool schema
# ------------------------------------------------------------------

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_similar_molecules",
        "description": (
            "Find the most similar molecules from the training set using task-aware "
            "embeddings. Returns K nearest neighbors with their labels, cached molecular "
            "properties, and shared scaffolds (MCS), plus the closest molecule with the "
            "opposite label for contrastive SAR reasoning. Use this for structure-activity analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the query molecule."},
                "task": {"type": "string", "description": "Task name for task-aware retrieval (e.g., 'AMES', 'DILI')."},
                "k": {"type": "integer", "description": "Number of nearest neighbors to retrieve (default: 5)."}
            },
            "required": ["smiles", "task"],
            "additionalProperties": False,
        }
    }
}
