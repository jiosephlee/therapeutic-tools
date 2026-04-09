"""
Version 10 therapeutic tool consolidation.

Adds two higher-level tools for downstream tool-calling RL:
  - get_molecular_properties: combines the existing molecule-analysis tools
  - get_similar_neighbors: returns a minimal neighbor-only similarity view
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .similarity import TASK_ALIASES, TASKS


def get_molecular_properties(smiles: str) -> str:
    """
    Return a consolidated molecular report by composing the existing tools.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string aggregating the existing property tools.
    """
    from .adme import assess_adme_properties
    from .functional_groups import analyze_functional_groups
    from .metabolism import predict_metabolites
    from .molecule_profile import get_molecule_profile
    from .ring_systems import analyze_ring_systems
    from .safety import screen_safety
    from .three_d import get_3d_properties

    sections = [
        get_molecule_profile(
            smiles,
            include_lipinski_violations=False,
            include_electronic_summary=False,
            include_quantum_properties=False,
        ),
        analyze_functional_groups(smiles, simple=True),
        analyze_ring_systems(smiles),
        assess_adme_properties(smiles, ph=7.4, simple_pka=True),
        get_3d_properties(smiles, include_epsa=False),
        screen_safety(smiles),
        predict_metabolites(smiles, max_metabolites=1),
    ]

    return "\n\n".join(section for section in sections if section)


def _label_str(label: Any) -> str:
    if label == 1:
        return "B"
    if label == 0:
        return "A"
    return str(label)


def _compute_local_knn_metrics(
    data: Dict[str, Any],
    embedding_type: str,
    display_k: int,
    top_neighborhood_idx: np.ndarray,
    all_train_mask: np.ndarray,
    exact_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Mirror the local leave-one-out neighborhood confidence from similarity.py."""
    train_labels = data["labels"]
    local_train: Dict[str, Optional[float]] = {"acc": None, "f1": None, "k": 0}
    if train_labels is None:
        return local_train

    from .similarity import _weighted_tanimoto

    train_only_idx = np.where(all_train_mask & ~exact_mask)[0]
    n_train_available = len(train_only_idx)
    neighborhood_k = 27
    eff_nbr_k = min(neighborhood_k, n_train_available)
    if eff_nbr_k < 3 or display_k <= 0:
        return local_train

    nbr_idx = top_neighborhood_idx[:eff_nbr_k]
    train_only_labels = train_labels[train_only_idx].astype(int)
    nbr_true: List[int] = []
    nbr_pred: List[int] = []

    if embedding_type == "learned":
        embeddings = data["embeddings"]
        train_only_embs = embeddings[train_only_idx]
        t_norms = np.linalg.norm(train_only_embs, axis=1, keepdims=True) + 1e-8
        train_only_normed = train_only_embs / t_norms

        for ni in nbr_idx:
            n_emb = embeddings[ni]
            n_norm = n_emb / (np.linalg.norm(n_emb) + 1e-8)
            n_sims = train_only_normed @ n_norm
            self_pos = np.where(train_only_idx == ni)[0]
            if len(self_pos):
                n_sims[self_pos[0]] = -np.inf
            top_voters = np.argsort(n_sims)[::-1][:display_k]
            voter_labels = train_only_labels[top_voters]
            nbr_true.append(int(train_labels[ni]))
            nbr_pred.append(int(np.round(voter_labels.mean())))
    else:
        morgan_fps = data["morgan_fps"]
        feat_fps = data["feat_fps"]
        train_only_morgans = morgan_fps[train_only_idx]
        train_only_feats = feat_fps[train_only_idx]

        for ni in nbr_idx:
            n_sims = _weighted_tanimoto(
                morgan_fps[ni],
                feat_fps[ni],
                train_only_morgans,
                train_only_feats,
            )
            self_pos = np.where(train_only_idx == ni)[0]
            if len(self_pos):
                n_sims[self_pos[0]] = -np.inf
            top_voters = np.argsort(n_sims)[::-1][:display_k]
            voter_labels = train_only_labels[top_voters]
            nbr_true.append(int(train_labels[ni]))
            nbr_pred.append(int(np.round(voter_labels.mean())))

    if nbr_true:
        local_train["acc"] = accuracy_score(nbr_true, nbr_pred)
        local_train["f1"] = f1_score(nbr_true, nbr_pred, average="macro", zero_division=0)
        local_train["k"] = eff_nbr_k
    return local_train


def get_similar_neighbors(
    smiles: str,
    task: str,
    k: int = 5,
    embedding_type: str = "fingerprint",
) -> str:
    """
    Return a minimal similarity report with neighbors and one contrastive example.

    Unlike ``find_similar_molecules``, this intentionally omits per-neighbor
    property and functional-group detail so the model can call
    ``get_molecular_properties`` on neighbors it wants to inspect further.
    """
    from .similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _cosine_similarities,
        _load_knn_metrics,
        _load_split_smiles,
        _load_task_data,
        _weighted_tanimoto,
    )

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if embedding_type not in {"learned", "fingerprint"}:
        raise ValueError("embedding_type must be 'learned' or 'fingerprint'")

    data = _load_task_data(task, embedding_type)
    if data is None:
        return (
            f"Error: No precomputed {embedding_type} embeddings found for task '{task}'.\n"
            f"Expected: cache/{embedding_type}/{task}_embeddings.npz"
        )

    train_smiles = data["smiles"]
    train_labels = data["labels"]
    match_idx = np.where(train_smiles == smiles)[0]
    exact_mask = train_smiles == smiles
    cached_canonical_smiles = data.get("canonical_smiles")

    if embedding_type == "fingerprint" and len(match_idx) == 0 and cached_canonical_smiles is not None:
        query_canonical_smiles = _canonicalize_smiles(smiles)
        if query_canonical_smiles is not None:
            canonical_mask = cached_canonical_smiles == query_canonical_smiles
            canonical_match_idx = np.where(canonical_mask)[0]
            if len(canonical_match_idx) > 0:
                match_idx = canonical_match_idx
                exact_mask = exact_mask | canonical_mask

    if embedding_type == "learned":
        if len(match_idx) == 0:
            return (
                f"Error: Query molecule '{smiles}' not found in the "
                f"precomputed {embedding_type} embeddings for task '{task}'."
            )
        embeddings = data["embeddings"]
        query_emb = embeddings[match_idx[0]]
        similarities = _cosine_similarities(query_emb, embeddings)
    else:
        morgan_fps = data["morgan_fps"]
        feat_fps = data["feat_fps"]
        if len(match_idx) > 0:
            query_morgan = morgan_fps[match_idx[0]]
            query_feat = feat_fps[match_idx[0]]
        else:
            query_morgan = _compute_query_fp(smiles, use_features=False)
            query_feat = _compute_query_fp(smiles, use_features=True)
            if query_morgan is None or query_feat is None:
                return f"Error: Could not compute fingerprint for query '{smiles}'."
        similarities = _weighted_tanimoto(query_morgan, query_feat, morgan_fps, feat_fps)

    similarities[exact_mask] = -np.inf

    splits = data.get("splits")
    if splits is not None:
        all_train_mask = splits == "train"
    else:
        split_data = _load_split_smiles(task)
        if split_data is not None:
            train_smi_set = split_data["train"]
            all_train_mask = np.array([s in train_smi_set for s in train_smiles])
        else:
            all_train_mask = np.ones(len(train_smiles), dtype=bool)

    train_sims = similarities.copy()
    train_sims[~all_train_mask] = -np.inf

    neighborhood_k = 27
    n_train_available = int((all_train_mask & ~exact_mask).sum())
    effective_neighborhood_k = min(neighborhood_k, n_train_available)
    display_k = min(k, n_train_available)

    top_neighborhood_idx = np.argsort(train_sims)[::-1][:effective_neighborhood_k]
    top_k_idx = top_neighborhood_idx[:display_k]

    neighbors: List[Dict[str, Any]] = []
    for idx in top_k_idx:
        neighbors.append(
            {
                "smiles": str(train_smiles[idx]),
                "similarity": float(similarities[idx]),
                "label": int(train_labels[idx]) if train_labels is not None else "N/A",
            }
        )

    local_train = _compute_local_knn_metrics(
        data=data,
        embedding_type=embedding_type,
        display_k=display_k,
        top_neighborhood_idx=top_neighborhood_idx,
        all_train_mask=all_train_mask,
        exact_mask=exact_mask,
    )

    contrastive = None
    if train_labels is not None:
        if len(match_idx) > 0:
            query_label = int(train_labels[match_idx[0]])
        elif neighbors:
            query_label = neighbors[0]["label"]
        else:
            query_label = None

        if query_label is not None:
            opposite_mask = (train_labels != query_label) & ~exact_mask & all_train_mask
            if np.any(opposite_mask):
                contra_sims = similarities.copy()
                contra_sims[~opposite_mask] = -np.inf
                contra_idx = int(np.argmax(contra_sims))
                contrastive = {
                    "smiles": str(train_smiles[contra_idx]),
                    "similarity": float(similarities[contra_idx]),
                    "label": int(train_labels[contra_idx]),
                }

    emb_label = "fingerprint (weighted Tanimoto)" if embedding_type == "fingerprint" else "learned embeddings"
    sections = [f"Similar Neighbors for task '{task}' based on {emb_label}", ""]

    all_metrics = _load_knn_metrics()
    knn_metrics = all_metrics.get(embedding_type, all_metrics).get(task, {})
    train_acc = knn_metrics.get("train_accuracy")
    train_f1 = knn_metrics.get("train_f1")
    has_local = local_train.get("acc") is not None
    if train_acc is not None or has_local:
        sections.append("KNN Metrics (train, leave-one-out):")
        if train_acc is not None:
            sections.append(f"- global -> accuracy={train_acc:.3f}, F1={train_f1:.3f}")
        if has_local:
            sections.append(
                f"- local ({local_train['k']} nearest neighbors) -> "
                f"accuracy={local_train['acc']:.3f}, F1={local_train['f1']:.3f}"
            )
        sections.append("")

    sections.append("Nearest Neighbors from Training Set:")
    if neighbors:
        for i, neighbor in enumerate(neighbors, 1):
            sections.append(
                f"{i}. {neighbor['smiles']} "
                f"(similarity: {neighbor['similarity']:.4f}, label: {_label_str(neighbor['label'])})"
            )
    else:
        sections.append("None found.")

    sections.append("")
    sections.append("Contrastive Example:")
    if contrastive is None:
        sections.append("None found.")
    else:
        sections.append(
            f"- {contrastive['smiles']} "
            f"(similarity: {contrastive['similarity']:.4f}, label: {_label_str(contrastive['label'])})"
        )

    return "\n".join(sections)


GET_MOLECULAR_PROPERTIES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_molecular_properties",
        "description": (
            "Returns molecular properties including molecular weight, TPSA, LogP, functional groups, ring systems, pka, 3D shape, "
            "structural alerts, and the top predicted metabolite."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Input molecule as a SMILES string."},
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}


GET_SIMILAR_NEIGHBORS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_similar_neighbors",
        "description": (
            "Find nearest training-set neighbors and one contrastive example. "
            "Returns only SMILES, similarity scores, labels, and KNN confidence metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Input molecule as a SMILES string."},
                "task": {"type": "string", "description": "Task name (for example 'AMES' or 'DILI')."},
                "k": {"type": "integer", "description": "Number of nearest neighbors to return. Default 5."},
                "embedding_type": {
                    "type": "string",
                    "description": "Similarity backend to use: learned embeddings or fingerprint Tanimoto.",
                    "enum": ["learned", "fingerprint"],
                },
            },
            "required": ["smiles", "task"],
            "additionalProperties": False,
        },
    },
}


def _task_alias(task: str) -> str:
    """Return the short alias for *task*, falling back to the lowercased name."""
    return TASK_ALIASES.get(task, task.lower())


def _make_task_neighbors_tool_schema(task: str) -> Dict[str, Any]:
    """Generate a task-specific v10 neighbors schema with task baked in."""
    alias = _task_alias(task)
    return {
        "type": "function",
        "function": {
            "name": f"get_similar_neighbors_{alias}",
            "description": (
                f"Find nearest neighbors from the {task} training set with labels, "
                "similarity scores, KNN confidence metrics, and one contrastive example."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "Input molecule as a SMILES string."},
                    "k": {
                        "type": "integer",
                        "description": "Number of nearest neighbors to return. Default 5.",
                    },
                },
                "required": ["smiles"],
                "additionalProperties": False,
            },
        },
    }


def _make_task_neighbors_callable(task: str):
    """Return a callable that calls get_similar_neighbors with task pre-filled."""
    alias = _task_alias(task)

    def _get_similar_neighbors(smiles: str, k: int = 5) -> str:
        return get_similar_neighbors(smiles, task=task, k=k, embedding_type="fingerprint")

    _get_similar_neighbors.__name__ = f"get_similar_neighbors_{alias}"
    return _get_similar_neighbors


TASK_NEIGHBOR_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    task: _make_task_neighbors_tool_schema(task) for task in TASKS
}


TASK_NEIGHBOR_CALLABLES: Dict[str, Any] = {
    f"get_similar_neighbors_{_task_alias(task)}": _make_task_neighbors_callable(task) for task in TASKS
}
