"""
Tool 7: Similar Molecules — task-aware embedding retrieval + MCS + contrastive example.

Supports two embedding types:
  - "learned"      (default): uses precomputed neural/AttNSOM embeddings, cosine similarity.
  - "fingerprint"  : uses Morgan + FeatureMorgan bit vectors, weighted Tanimoto similarity
                     (0.8 × morgantanimoto + 0.2 × feat_morgan tanimoto), matching the
                     Intern-S1-recipe KNN baseline.

Cache layout:
  cache/learned/{task}_embeddings.npz      — smiles, embeddings, labels
  cache/<fingerprint_subdir>/{task}_embeddings.npz
      — smiles, canonical_smiles, morgan_fps, feat_morgan_fps, labels
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from functools import lru_cache
from sklearn.metrics import accuracy_score, f1_score

import json as _json

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "tdc", "raw")

EMBEDDING_TYPES = ("learned", "fingerprint")
_FINGERPRINT_CACHE_CANDIDATES = (
    os.environ.get("THERAPEUTIC_FINGERPRINT_CACHE_SUBDIR"),
    "fingerprints_with_canonicalized",
    "fingerprint_v8",
    "fingerprint",
)


# ------------------------------------------------------------------
# Embedding & metadata loaders
# ------------------------------------------------------------------

def _embeddings_path(task: str, embedding_type: str = "learned") -> str:
    if embedding_type != "fingerprint":
        return os.path.join(_CACHE_DIR, embedding_type, f"{task}_embeddings.npz")

    for subdir in _FINGERPRINT_CACHE_CANDIDATES:
        if not subdir:
            continue
        candidate = os.path.join(_CACHE_DIR, subdir, f"{task}_embeddings.npz")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_CACHE_DIR, "fingerprint", f"{task}_embeddings.npz")


@lru_cache(maxsize=64)
def _load_task_data(task: str, embedding_type: str = "learned"):
    """Load precomputed embeddings for *task* and *embedding_type*.

    Returns a dict with type-specific keys, or ``None`` if the file is missing.

    For ``"learned"``:
        {"type": "learned", "smiles": ..., "embeddings": ..., "labels": ...}
    For ``"fingerprint"``:
        {"type": "fingerprint", "smiles": ..., "morgan_fps": ..., "feat_morgan_fps": ..., "labels": ...}
    """
    npz_path = _embeddings_path(task, embedding_type)
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)

    if embedding_type == "learned":
        smiles = d["smiles"]
        embeddings = d["embeddings"].astype(np.float32)
        labels = d["labels"] if "labels" in d else None
        return {"type": "learned", "smiles": smiles, "embeddings": embeddings, "labels": labels}

    elif embedding_type == "fingerprint":
        smiles = d["smiles"]
        canonical_smiles = d["canonical_smiles"] if "canonical_smiles" in d else None
        morgan_fps = d["morgan_fps"].astype(np.float32)       # (N, 2048)
        feat_fps   = d["feat_morgan_fps"].astype(np.float32)  # (N, 2048)
        labels = d["labels"] if "labels" in d else None
        splits = d["splits"] if "splits" in d else None
        return {"type": "fingerprint", "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "morgan_fps": morgan_fps, "feat_fps": feat_fps,
                "labels": labels, "splits": splits}

    return None


def _load_metadata() -> Optional[pd.DataFrame]:
    """Delegate to shared metadata_cache module."""
    from . import metadata_cache
    return metadata_cache._load_metadata()


@lru_cache(maxsize=32)
def _load_split_smiles(task: str) -> Optional[dict]:
    """Load train/val SMILES sets for *task* from raw TDC data.

    Returns dict with 'train' and 'val' sets, or None if files missing.
    Tries both data/tdc/raw and data/TDC paths.
    """
    for base in [_DATA_DIR, os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "TDC")]:
        train_path = os.path.join(base, task, "train.csv")
        val_path = os.path.join(base, task, "val.csv")
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_smi = set(pd.read_csv(train_path)["Drug"].dropna())
            val_smi = set(pd.read_csv(val_path)["Drug"].dropna())
            return {"train": train_smi, "val": val_smi}
    return None


@lru_cache(maxsize=1)
def _load_knn_metrics() -> dict:
    """Load precomputed KNN accuracy/F1 per task and embedding type.

    Returns nested dict: {embedding_type: {task: metrics}}.
    Falls back to flat {task: metrics} format for backward compatibility.
    """
    path = os.path.join(_CACHE_DIR, "knn_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            data = _json.load(f)
        # Detect new nested format
        if any(k in data for k in EMBEDDING_TYPES):
            return data
        # Legacy flat format — treat as "learned"
        return {"learned": data}
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
# Similarity computation
# ------------------------------------------------------------------

def _cosine_similarities(query_emb: np.ndarray, all_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity of *query_emb* against every row of *all_embs*."""
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
    return (all_embs / norms) @ q_norm  # (N,)


def _tanimoto_similarities(query_fp: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """Bulk Tanimoto similarity of a single query FP against *ref_fps*.

    Both query_fp and ref_fps should be float32 arrays with values in {0, 1}.
    Tanimoto = |A & B| / |A | B| = dot(A, B) / (|A| + |B| - dot(A, B))
    """
    a_and_b = ref_fps @ query_fp                   # (N,)
    a_bits = query_fp.sum()
    b_bits = ref_fps.sum(axis=1)                   # (N,)
    denom = a_bits + b_bits - a_and_b
    # Avoid divide-by-zero for empty fingerprints
    return np.where(denom > 0, a_and_b / denom, 0.0)


def _weighted_tanimoto(
    query_morgan: np.ndarray,
    query_feat: np.ndarray,
    ref_morgans: np.ndarray,
    ref_feats: np.ndarray,
    w_morgan: float = 0.8,
    w_feat: float = 0.2,
) -> np.ndarray:
    """Weighted Tanimoto similarity matching the Intern-S1 KNN formula."""
    t_morgan = _tanimoto_similarities(query_morgan, ref_morgans)
    t_feat   = _tanimoto_similarities(query_feat,   ref_feats)
    return w_morgan * t_morgan + w_feat * t_feat


def _compute_query_fp(smiles: str, use_features: bool = False) -> Optional[np.ndarray]:
    """Compute a Morgan fingerprint on-the-fly for a query not in the cache.

    Uses the same canonicalization as ``build_fingerprint_embeddings.py``
    so the fingerprint is comparable to the cached reference fingerprints.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        cleaned = Chem.MolFromSmiles(canonical_smiles)
        if use_features:
            inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        else:
            inv_gen = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, atomInvariantsGenerator=inv_gen,
        )
        fp = gen.GetFingerprintAsNumPy(cleaned)
        return fp.astype(np.float32)
    except Exception:
        return None


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string for cache lookup."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ------------------------------------------------------------------
# Core retrieval
# ------------------------------------------------------------------

def find_similar_molecules(
    smiles: str,
    task: str,
    k: int = 5,
    embedding_type: str = "learned",
    **kwargs,
) -> str:
    """
    Find similar molecules using task-aware embeddings.

    Uses precomputed embeddings to retrieve the K nearest neighbors
    from the training set, plus the closest molecule with the opposite label
    (contrastive example). Computes MCS between the query and each neighbor.

    Args:
        smiles: SMILES string of the query molecule.
        task: Task name (e.g., "AMES", "DILI", "BBB_Martins") for task-aware retrieval.
        k: Number of nearest neighbors to retrieve (default: 5).
        embedding_type: "learned" (default, cosine similarity) or "fingerprint"
            (weighted Tanimoto: 0.8×Morgan + 0.2×FeatureMorgan).
    Returns:
        Multi-line formatted string with similar molecules, their labels,
        metadata properties, and shared scaffolds (MCS).
    """
    # Internal-only compatibility flag used by dataset build scripts.
    include_pseudo_label = kwargs.pop(
        "include_pseudo_label",
        kwargs.pop("include_psuedo_label", False),
    )

    data = _load_task_data(task, embedding_type)
    if data is None:
        return (
            f"Error: No precomputed {embedding_type} embeddings found for task '{task}'.\n"
            f"Expected: cache/{embedding_type}/{task}_embeddings.npz"
        )

    train_smiles = data["smiles"]
    train_labels = data["labels"]

    # --- look up query embedding (or compute on-the-fly for fingerprint) ---
    match_idx = np.where(train_smiles == smiles)[0]
    exact_mask = (train_smiles == smiles)
    cached_canonical_smiles = data.get("canonical_smiles")

    if embedding_type == "fingerprint" and len(match_idx) == 0 and cached_canonical_smiles is not None:
        query_canonical_smiles = _canonicalize_smiles(smiles)
        if query_canonical_smiles is not None:
            canonical_mask = (cached_canonical_smiles == query_canonical_smiles)
            canonical_match_idx = np.where(canonical_mask)[0]
            if len(canonical_match_idx) > 0:
                match_idx = canonical_match_idx
                exact_mask = exact_mask | canonical_mask

    # --- compute similarities ---
    if embedding_type == "learned":
        if len(match_idx) == 0:
            return (
                f"Error: Query molecule '{smiles}' not found in the "
                f"precomputed {embedding_type} embeddings for task '{task}'."
            )
        embeddings = data["embeddings"]
        query_emb = embeddings[match_idx[0]]
        similarities = _cosine_similarities(query_emb, embeddings)

    elif embedding_type == "fingerprint":
        morgan_fps = data["morgan_fps"]
        feat_fps   = data["feat_fps"]
        if len(match_idx) > 0:
            query_morgan = morgan_fps[match_idx[0]]
            query_feat   = feat_fps[match_idx[0]]
        else:
            # Query not in embeddings (e.g. ambiguous label excluded it).
            # Compute fingerprint on the fly so we can still retrieve neighbors.
            query_morgan = _compute_query_fp(smiles, use_features=False)
            query_feat   = _compute_query_fp(smiles, use_features=True)
            if query_morgan is None or query_feat is None:
                return (
                    f"Error: Could not compute fingerprint for query '{smiles}'."
                )
        similarities = _weighted_tanimoto(query_morgan, query_feat, morgan_fps, feat_fps)

    else:
        return f"Error: Unknown embedding_type '{embedding_type}'. Use 'learned' or 'fingerprint'."

    # Mask self
    similarities[exact_mask] = -np.inf

    # --- Restrict neighbors to training split only ---
    splits = data.get("splits")
    if splits is not None:
        all_train_mask = (splits == "train")
        all_val_mask   = (splits == "val")
    else:
        # Legacy fallback: no splits in npz, try raw CSVs
        split_data = _load_split_smiles(task)
        if split_data is not None:
            train_smi_set = split_data["train"]
            val_smi_set   = split_data["val"]
            all_train_mask = np.array([s in train_smi_set for s in train_smiles])
            all_val_mask   = np.array([s in val_smi_set   for s in train_smiles])
        else:
            all_train_mask = np.ones(len(train_smiles), dtype=bool)
            all_val_mask   = np.zeros(len(train_smiles), dtype=bool)

    # Similarities restricted to train-only for neighbor retrieval
    train_sims = similarities.copy()
    train_sims[~all_train_mask] = -np.inf

    # Retrieve k=27 train neighbors for neighborhood confidence, display top-k
    neighborhood_k = 27
    n_train_available = int((all_train_mask & ~exact_mask).sum())
    effective_neighborhood_k = min(neighborhood_k, n_train_available)
    display_k = min(k, n_train_available)

    top_neighborhood_idx = np.argsort(train_sims)[::-1][:effective_neighborhood_k]
    top_k_idx = top_neighborhood_idx[:display_k]

    neighbors: List[dict] = []
    for idx in top_k_idx:
        neighbors.append({
            "smiles": str(train_smiles[idx]),
            "similarity": float(similarities[idx]),
            "label": int(train_labels[idx]) if train_labels is not None else "N/A",
        })

    # --- Local train neighborhood confidence ---
    # For the k=27 nearest train neighbors of the query, compute leave-one-out
    # KNN accuracy using the same display_k. This tells the model how reliable
    # KNN predictions are in this region of chemical space.
    local_train = {"acc": None, "f1": None, "k": 0}

    if train_labels is not None:
        train_only_idx = np.where(all_train_mask & ~exact_mask)[0]
        n_train_available = len(train_only_idx)
        eff_nbr_k = min(neighborhood_k, n_train_available)

        if eff_nbr_k >= 3:
            # Find the k=27 nearest train neighbors of the query
            nbr_idx = top_neighborhood_idx[:eff_nbr_k]
            train_only_labels = train_labels[train_only_idx].astype(int)

            # For each neighbor, predict its label using its own k nearest
            # train neighbors (excluding itself)
            nbr_true = []
            nbr_pred = []

            if embedding_type == "learned":
                embeddings = data["embeddings"]
                train_only_embs = embeddings[train_only_idx]
                t_norms = np.linalg.norm(train_only_embs, axis=1, keepdims=True) + 1e-8
                train_only_normed = train_only_embs / t_norms

                for ni in nbr_idx:
                    n_emb = embeddings[ni]
                    n_norm = n_emb / (np.linalg.norm(n_emb) + 1e-8)
                    n_sims = train_only_normed @ n_norm
                    # Exclude self
                    self_pos = np.where(train_only_idx == ni)[0]
                    if len(self_pos):
                        n_sims[self_pos[0]] = -np.inf
                    top_voters = np.argsort(n_sims)[::-1][:display_k]
                    voter_labels = train_only_labels[top_voters]
                    nbr_true.append(int(train_labels[ni]))
                    nbr_pred.append(int(np.round(voter_labels.mean())))

            elif embedding_type == "fingerprint":
                morgan_fps = data["morgan_fps"]
                feat_fps   = data["feat_fps"]
                train_only_morgans = morgan_fps[train_only_idx]
                train_only_feats   = feat_fps[train_only_idx]

                for ni in nbr_idx:
                    n_sims = _weighted_tanimoto(
                        morgan_fps[ni], feat_fps[ni],
                        train_only_morgans, train_only_feats,
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
                local_train["f1"]  = f1_score(nbr_true, nbr_pred, average='macro', zero_division=0)
                local_train["k"]   = eff_nbr_k

    # --- contrastive example (closest opposite-label molecule) ---
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

    return _format_results(
        smiles, task, neighbors, contrastive,
        local_train=local_train,
        embedding_type=embedding_type,
        include_pseudo_label=include_pseudo_label,
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
        names = []
        for line in desc.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                name_part = line[2:].split(":")[0].strip()
                names.append(name_part)
        return ", ".join(names) if names else None
    except Exception:
        return None


def _label_str(label) -> str:
    """Convert numeric label to A/B string."""
    if label == 1:
        return "B"
    elif label == 0:
        return "A"
    return str(label)


def _get_key_properties(smiles: str) -> Optional[str]:
    """Return MW, logP, TPSA for a neighbor molecule. Compute if not cached."""
    from . import metadata_cache
    cached = metadata_cache.lookup_row(smiles)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if cached and "MolWt" in cached:
            mw = cached["MolWt"]
        else:
            mw = Descriptors.MolWt(mol)

        if cached and "MolLogP" in cached:
            logp = cached["MolLogP"]
        else:
            logp = Crippen.MolLogP(mol)

        if cached and "TPSA" in cached:
            tpsa = cached["TPSA"]
        else:
            tpsa = rdMolDescriptors.CalcTPSA(mol)

        return f"MW={mw:.1f}, logP={logp:.2f}, TPSA={tpsa:.1f}"
    except Exception:
        return None


def _format_neighbor(idx: int, neighbor: dict) -> str:
    """Format a single neighbor for output."""
    label_str = _label_str(neighbor['label'])
    line = (
        f"{idx}. {neighbor['smiles']} "
        f"(similarity: {neighbor['similarity']:.4f}, label: {label_str})"
    )
    props = _get_key_properties(neighbor['smiles'])
    if props:
        line += f"\n   Basic properties: {props}"
    fg = _get_fg_summary(neighbor['smiles'])
    if fg:
        line += f"\n   Functional groups: {fg}"
    return line


def _format_results(
    query_smiles: str,
    task: str,
    neighbors: List[dict],
    contrastive: Optional[dict],
    local_train: Optional[dict] = None,
    embedding_type: str = "learned",
    include_pseudo_label: bool = False,
) -> str:
    """Format the full similarity search output."""
    emb_label = "fingerprint (weighted Tanimoto)" if embedding_type == "fingerprint" else "learned embeddings"
    sections = [f"Similar Molecules for task '{task}' based on {emb_label}", ""]

    # KNN metrics (train split)
    all_metrics = _load_knn_metrics()
    knn_metrics = all_metrics.get(embedding_type, all_metrics).get(task, {})
    train_acc = knn_metrics.get("train_accuracy")
    train_f1  = knn_metrics.get("train_f1")
    has_local = local_train and local_train.get("acc") is not None

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
    for i, neighbor in enumerate(neighbors, 1):
        sections.append(_format_neighbor(i, neighbor))

    # Only show contrastive example if its label is not already in the top-k
    if contrastive:
        top_k_labels = {n["label"] for n in neighbors}
        if contrastive["label"] not in top_k_labels:
            sections.append("")
            sections.append("Nearest Contrastive Example (opposite label):")
            sections.append(_format_neighbor(len(neighbors) + 1, contrastive))

    # Append KNN pseudo-label (majority vote of displayed neighbors)
    if include_pseudo_label and neighbors:
        neighbor_labels = [n["label"] for n in neighbors]
        mean_label = sum(neighbor_labels) / len(neighbor_labels)
        pseudo = 1 if mean_label >= 0.5 else 0
        label_str = "(B)" if pseudo == 1 else "(A)"
        sections.append("")
        sections.append(f"KNN Predicted Label: {label_str}")

    return "\n".join(sections)


# ------------------------------------------------------------------
# Tool schema (generic, requires task parameter)
# ------------------------------------------------------------------

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_similar_molecules",
        "description": "Find K nearest neighbors from training set with labels, properties, and contrastive opposite-label example.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "task": {"type": "string", "description": "Task name (e.g. 'AMES', 'DILI')."},
                "k": {"type": "integer", "description": "Number of neighbors (default: 5)."},
                "embedding_type": {
                    "type": "string",
                    "enum": ["learned", "fingerprint"],
                },
            },
            "required": ["smiles", "task"],
        }
    }
}


# ------------------------------------------------------------------
# Per-task tool schemas and dispatch
# ------------------------------------------------------------------

# All TDC tasks with fingerprint embeddings
TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin", "ClinTox",
    "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG",
]

# Short aliases for task-specific function names.
# Long TDC task names (e.g. CYP2C9_Substrate_CarbonMangels) are hard for
# LLMs to reproduce exactly, so we use concise proxies as the tool name
# the model sees.  Tasks not listed here use their original name lowercased.
TASK_ALIASES: Dict[str, str] = {
    "AMES": "ames",
    "BBB_Martins": "bbb",
    "Bioavailability_Ma": "bioavail",
    "CYP2C9_Substrate_CarbonMangels": "cyp2c9",
    "CYP2D6_Substrate_CarbonMangels": "cyp2d6",
    "CYP3A4_Substrate_CarbonMangels": "cyp3a4",
    "Carcinogens_Lagunin": "carcinogens",
    "ClinTox": "clintox",
    "DILI": "dili",
    "HIA_Hou": "hia",
    "PAMPA_NCATS": "pampa",
    "Pgp_Broccatelli": "pgp",
    "SARSCoV2_3CLPro_Diamond": "sarscov2_3cl",
    "SARSCoV2_Vitro_Touret": "sarscov2_vitro",
    "Skin_Reaction": "skin",
    "hERG": "herg",
}


def _task_alias(task: str) -> str:
    """Return the short alias for *task*, falling back to the lowercased name."""
    return TASK_ALIASES.get(task, task.lower())


def _make_task_tool_schema(task: str) -> Dict[str, Any]:
    """Generate a task-specific tool schema with task baked in."""
    alias = _task_alias(task)
    return {
        "type": "function",
        "function": {
            "name": f"find_similar_molecules_{alias}",
            "description": f"Find nearest neighbors from {task} training set with labels and contrastive example.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                },
                "required": ["smiles"],
            }
        }
    }


def _make_task_callable(task: str):
    """Return a callable that calls find_similar_molecules with task pre-filled."""
    alias = _task_alias(task)
    def _find_similar(smiles: str, **kwargs) -> str:
        include_pseudo_label = kwargs.pop(
            "include_pseudo_label",
            kwargs.pop("include_psuedo_label", False),
        )
        return find_similar_molecules(
            smiles, task=task, k=5, embedding_type="fingerprint",
            include_pseudo_label=include_pseudo_label,
        )
    _find_similar.__name__ = f"find_similar_molecules_{alias}"
    return _find_similar


# Pre-built registries for all tasks
TASK_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    task: _make_task_tool_schema(task) for task in TASKS
}

TASK_CALLABLES: Dict[str, Any] = {
    f"find_similar_molecules_{_task_alias(task)}": _make_task_callable(task) for task in TASKS
}
