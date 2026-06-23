"""
Version 11 therapeutic tools — generalized get_features + get_neighbors.

Two tools with a shared feature registry:
  - get_features(smiles, feature_names): selective feature computation
  - get_neighbors_{task}(smiles, ...): per-task nearest neighbors with inline features

Feature names are advertised in the system prompt, NOT in the tool JSON schema.
Both feature names and task names support case-insensitive fuzzy matching
(Levenshtein distance <= 2).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.similarity import TASKS, TASK_ALIASES

# ---------------------------------------------------------------------------
# Feature registry — each entry maps a name to a callable(smiles) -> str
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "physicochemical",
    "complexity",
    "electronic",
    "functional_groups",
    "ring_systems",
    "3d_properties",
    "structural_alerts",
    "metabolites",
    "pka",
    "ionization",
    "logd",
    "solubility",
]


# ---------------------------------------------------------------------------
# Fuzzy matching (Levenshtein distance <= 2, case-insensitive)
# ---------------------------------------------------------------------------

def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _fuzzy_match(query: str, candidates: List[str], max_dist: int = 2) -> Optional[str]:
    """Match *query* against *candidates* (case-insensitive, Levenshtein <= max_dist).

    Returns the canonical candidate name or None if no match.
    Exact (case-insensitive) match takes priority; then best fuzzy match.
    """
    q = query.lower()
    # Exact match first
    for c in candidates:
        if q == c.lower():
            return c
    # Fuzzy match — pick closest
    best, best_dist = None, max_dist + 1
    for c in candidates:
        d = _levenshtein(q, c.lower())
        if d < best_dist:
            best, best_dist = c, d
    return best if best_dist <= max_dist else None


def _resolve_feature_names(raw_names: List[str]) -> tuple:
    """Resolve a list of user-supplied feature names via fuzzy matching.

    Returns (resolved_names: List[str], errors: List[str]).
    """
    resolved = []
    errors = []
    for name in raw_names:
        match = _fuzzy_match(name, FEATURE_NAMES)
        if match is not None:
            resolved.append(match)
        else:
            errors.append(name)
    return resolved, errors


def _resolve_task_name(raw_task: str) -> Optional[str]:
    """Resolve a user-supplied task name via fuzzy matching against TASKS."""
    return _fuzzy_match(raw_task, TASKS)


def _compute_physicochemical(smiles: str) -> str:
    """MW, logP, TPSA, HBD/HBA, rotatable bonds, Fsp3, molar refractivity."""
    from ..utils import metadata_cache
    from ..utils.molecule_profile import _mol_from_smiles

    cached = metadata_cache.lookup_row(smiles)

    def _c(prop, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen

    mol = _mol_from_smiles(smiles)

    mw = _c("MolWt", lambda: float(Descriptors.MolWt(mol)))
    heavy_atoms = int(_c("HeavyAtomCount", lambda: float(Descriptors.HeavyAtomCount(mol))))
    heteroatoms = int(_c("NumHeteroatoms", lambda: float(Lipinski.NumHeteroatoms(mol))))
    logp = _c("MolLogP", lambda: float(Crippen.MolLogP(mol)))
    tpsa = _c("TPSA", lambda: float(rdMolDescriptors.CalcTPSA(mol)))
    hbd = int(_c("NumHDonors", lambda: float(Lipinski.NumHDonors(mol))))
    hba = int(_c("NumHAcceptors", lambda: float(Lipinski.NumHAcceptors(mol))))
    rotatable = int(_c("NumRotatableBonds", lambda: float(Lipinski.NumRotatableBonds(mol))))
    fsp3 = _c("FractionCSP3", lambda: float(rdMolDescriptors.CalcFractionCSP3(mol)))
    mr = _c("MolMR", lambda: float(Crippen.MolMR(mol)))

    return "\n".join([
        "Physicochemical Properties:",
        f"- Molecular weight: {mw:.2f} Da",
        f"- Heavy atoms: {heavy_atoms}, Heteroatoms: {heteroatoms}",
        f"- logP (Wildman-Crippen): {logp:.2f}",
        f"- TPSA: {tpsa:.2f} \u00c5\u00b2",
        f"- H-bond donors: {hbd}, H-bond acceptors: {hba}",
        f"- Rotatable bonds: {rotatable}",
        f"- Fraction sp3 carbons (Fsp3): {fsp3:.2f}",
        f"- Molar refractivity: {mr:.2f}",
    ])


def _compute_complexity(smiles: str) -> str:
    """Bertz CT, Hall-Kier Alpha, Kappa1-3, Balaban J, amide bonds, stereocenters."""
    from ..utils import metadata_cache
    from ..utils.molecule_profile import _mol_from_smiles

    cached = metadata_cache.lookup_row(smiles)

    def _c(prop, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, GraphDescriptors

    mol = _mol_from_smiles(smiles)

    bertz = _c("BertzCT", lambda: float(Descriptors.BertzCT(mol)))
    hall_kier = _c("HallKierAlpha", lambda: float(Descriptors.HallKierAlpha(mol)))
    kappa1 = _c("Kappa1", lambda: float(GraphDescriptors.Kappa1(mol)))
    kappa2 = _c("Kappa2", lambda: float(GraphDescriptors.Kappa2(mol)))
    kappa3 = _c("Kappa3", lambda: float(GraphDescriptors.Kappa3(mol)))
    balaban_j = _c("BalabanJ", lambda: float(GraphDescriptors.BalabanJ(mol)))
    amide_bonds = int(_c("NumAmideBonds", lambda: float(Lipinski.NumAmideBonds(mol))))
    stereocenters = int(_c("NumAtomStereoCenters", lambda: float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))))

    return "\n".join([
        "Complexity Metrics:",
        f"- Bertz complexity: {bertz:.2f}",
        f"- Topology: HallKierAlpha={hall_kier:.2f}, Kappa1={kappa1:.2f}, Kappa2={kappa2:.2f}, Kappa3={kappa3:.2f}, BalabanJ={balaban_j:.2f}",
        f"- Amide bonds: {amide_bonds}",
        f"- Stereocenters: {stereocenters}",
    ])


def _compute_electronic(smiles: str) -> str:
    """Gasteiger charges, charge polarization, EState indices."""
    from ..utils import metadata_cache
    from ..utils.molecule_profile import _mol_from_smiles

    cached = metadata_cache.lookup_row(smiles)

    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem

    mol = _mol_from_smiles(smiles)

    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        try:
            q = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not np.isnan(q):
                charges.append((atom.GetIdx(), atom.GetSymbol(), q))
        except Exception:
            pass

    if charges:
        max_charge = max(charges, key=lambda x: x[2])
        min_charge = min(charges, key=lambda x: x[2])
        charge_polarization = max_charge[2] - min_charge[2]
    else:
        max_charge = (0, "?", float("nan"))
        min_charge = (0, "?", float("nan"))
        charge_polarization = float("nan")

    if cached and "MaxEStateIndex" in cached:
        max_estate = cached["MaxEStateIndex"]
    else:
        max_estate = float(Descriptors.MaxEStateIndex(mol))
    if cached and "MinEStateIndex" in cached:
        min_estate = cached["MinEStateIndex"]
    else:
        min_estate = float(Descriptors.MinEStateIndex(mol))

    return "\n".join([
        "Electronic Properties:",
        f"- Gasteiger charges: max={max_charge[2]:.2f} (atom {max_charge[0]}, {max_charge[1]}), min={min_charge[2]:.2f} (atom {min_charge[0]}, {min_charge[1]})",
        f"- Charge polarization: {charge_polarization:.2f}",
        f"- EState indices: max={max_estate:.2f} (nucleophilic proxy), min={min_estate:.2f} (electrophilic proxy)",
    ])


def _compute_pka(smiles: str) -> str:
    """pKa prediction via MolGpKa."""
    from ..utils import endpoints
    from ..utils.adme import _format_pka_section

    pka_data = endpoints.pka_summary(smiles)
    if pka_data is None:
        return f"pKa Summary:\n- Invalid SMILES: {smiles!r}"
    return _format_pka_section(smiles, pka_data, simple=True)


def _compute_ionization(smiles: str) -> str:
    """Ionization state at pH 7.4 via Dimorphite-DL."""
    from ..utils.adme import _compact_ionization

    try:
        return _compact_ionization(smiles, ph=7.4)
    except Exception as e:
        return f"Ionization Classification: Error - {e}"


def _compute_logd(smiles: str) -> str:
    """LogD at pH 7.4 from Henderson-Hasselbalch."""
    from ..utils import endpoints

    logd = endpoints.logd_74(smiles)
    if logd is None:
        return "LogD at pH 7.4: unavailable"
    return f"LogD at pH 7.4: {logd:.2f}"


def _compute_solubility(smiles: str) -> str:
    """Solubility prediction."""
    from ..utils import endpoints

    log_s = endpoints.solubility_log_s(smiles)
    if log_s is None:
        return "Solubility: unavailable"
    return f"Solubility: logS = {log_s:.2f} log(mol/L)"


# Map feature names to compute functions.
# Section-level features delegate to existing module functions via lazy imports.
FEATURE_REGISTRY: Dict[str, Any] = {
    "physicochemical": _compute_physicochemical,
    "complexity": _compute_complexity,
    "electronic": _compute_electronic,
    "functional_groups": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.utils.functional_groups", fromlist=["analyze_functional_groups"]
    ).analyze_functional_groups(s, simple=True),
    "ring_systems": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.utils.ring_systems", fromlist=["analyze_ring_systems"]
    ).analyze_ring_systems(s),
    "3d_properties": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.utils.three_d", fromlist=["get_3d_properties"]
    ).get_3d_properties(s, include_epsa=False),
    "structural_alerts": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.utils.safety", fromlist=["screen_safety"]
    ).screen_safety(s),
    "metabolites": lambda s: __import__(
        "openrlhf.tools.therapeutic_tools.utils.metabolism", fromlist=["predict_metabolites"]
    ).predict_metabolites(s, max_metabolites=1),
    "pka": _compute_pka,
    "ionization": _compute_ionization,
    "logd": _compute_logd,
    "solubility": _compute_solubility,
}


def _compute_features_for_smiles(smiles: str, feature_names: List[str]) -> str:
    """Run requested features for a single SMILES. Returns joined sections."""
    sections = []
    for name in feature_names:
        fn = FEATURE_REGISTRY.get(name)
        if fn is None:
            continue
        try:
            result = fn(smiles)
            if result:
                sections.append(result)
        except Exception as e:
            sections.append(f"{name}: Error - {e}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def get_features(smiles: str, feature_names: List[str]) -> str:
    """
    Compute selected molecular features for a SMILES string.

    Args:
        smiles: Input molecule as a SMILES string.
        feature_names: List of feature categories to compute (fuzzy-matched).

    Returns:
        Multi-line formatted string with the requested feature sections.
    """
    if not feature_names:
        return f"Error: feature_names is empty. Available: {', '.join(FEATURE_NAMES)}"

    resolved, errors = _resolve_feature_names(feature_names)
    if errors:
        return (
            f"Error: Unknown feature(s): {', '.join(errors)}. "
            f"Available: {', '.join(FEATURE_NAMES)}"
        )

    return _compute_features_for_smiles(smiles, resolved)


def _label_str(label: Any) -> str:
    if label == 1:
        return "B"
    if label == 0:
        return "A"
    return str(label)


K_NEIGHBORS = 3


def get_neighbors(
    smiles: str,
    task_name: str,
    feature_names: Optional[List[str]] = None,
    include_labels: bool = True,
) -> str:
    """
    Find nearest training-set neighbors with optional inline feature computation.

    Args:
        smiles: Input molecule as a SMILES string.
        task_name: TDC task name (e.g. 'AMES', 'DILI'). Fuzzy-matched.
        feature_names: Optional list of features to compute for each neighbor (fuzzy-matched).
        include_labels: Whether to include task labels for neighbors (default True).

    Returns:
        Multi-line formatted string with neighbor details and optional features.
    """
    from ..utils.similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
        _weighted_tanimoto,
    )

    # Resolve task name
    resolved_task = _resolve_task_name(task_name)
    if resolved_task is None:
        return (
            f"Error: Unknown task '{task_name}'. "
            f"Available tasks: {', '.join(TASKS)}"
        )

    # Validate and resolve feature names if provided
    resolved_features = None
    if feature_names:
        resolved_features, errors = _resolve_feature_names(feature_names)
        if errors:
            return (
                f"Error: Unknown feature(s): {', '.join(errors)}. "
                f"Available: {', '.join(FEATURE_NAMES)}"
            )

    # Load fingerprint data for the task
    data = _load_task_data(resolved_task, "fingerprint")
    if data is None:
        return (
            f"Error: No precomputed fingerprint embeddings found for task '{resolved_task}'.\n"
            f"Available tasks: {', '.join(TASKS)}"
        )

    train_smiles = data["smiles"]
    train_labels = data["labels"]
    morgan_fps = data["morgan_fps"]
    feat_fps = data["feat_fps"]

    # Find query in cache or compute on the fly
    match_idx = np.where(train_smiles == smiles)[0]
    exact_mask = train_smiles == smiles

    cached_canonical_smiles = data.get("canonical_smiles")
    if len(match_idx) == 0 and cached_canonical_smiles is not None:
        query_canonical = _canonicalize_smiles(smiles)
        if query_canonical is not None:
            canonical_mask = cached_canonical_smiles == query_canonical
            canonical_match_idx = np.where(canonical_mask)[0]
            if len(canonical_match_idx) > 0:
                match_idx = canonical_match_idx
                exact_mask = exact_mask | canonical_mask

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

    # Restrict to training split
    splits = data.get("splits")
    if splits is not None:
        train_mask = splits == "train"
    else:
        split_data = _load_split_smiles(resolved_task)
        if split_data is not None:
            train_smi_set = split_data["train"]
            train_mask = np.array([s in train_smi_set for s in train_smiles])
        else:
            train_mask = np.ones(len(train_smiles), dtype=bool)

    train_sims = similarities.copy()
    train_sims[~train_mask] = -np.inf

    n_available = int((train_mask & ~exact_mask).sum())
    k = min(K_NEIGHBORS, n_available)

    top_idx = np.argsort(train_sims)[::-1][:k]

    # Build output
    sections = [f"Nearest Neighbors for task '{resolved_task}' (k={k}):"]

    for i, idx in enumerate(top_idx, 1):
        nbr_smiles = str(train_smiles[idx])
        sim = float(similarities[idx])
        label_part = ""
        if include_labels and train_labels is not None:
            label_part = f", label: {_label_str(int(train_labels[idx]))}"

        sections.append(f"\n{i}. {nbr_smiles} (similarity: {sim:.2f}{label_part})")

        if resolved_features:
            feat_text = _compute_features_for_smiles(nbr_smiles, resolved_features)
            if feat_text:
                # Indent feature output under the neighbor
                indented = "\n".join("   " + line for line in feat_text.split("\n"))
                sections.append(indented)

    if k == 0:
        sections.append("No neighbors found.")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

GET_FEATURES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_features",
        "description": (
            "Compute selected molecular features for a given SMILES string. "
            "Pass a list of feature names to select which analyses to run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Input molecule as a SMILES string.",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature categories to compute.",
                },
            },
            "required": ["smiles", "feature_names"],
            "additionalProperties": False,
        },
    },
}

GET_NEIGHBORS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_neighbors",
        "description": (
            "Retrieve close analogs for a molecule within a specified prediction task. "
            "Use this to compare the query molecule against similar compounds, inspect "
            "their labels, and optionally review selected feature groups for each analog."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Query molecule, provided as a SMILES string.",
                },
                "task_name": {
                    "type": "string",
                    "description": "Prediction task or assay context, for example AMES, DILI, or hERG.",
                },
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional feature groups to compute for each retrieved analog.",
                },
                "include_labels": {
                    "type": "boolean",
                    "description": "Whether to include observed task labels for the retrieved analogs. Default true.",
                },
            },
            "required": ["smiles", "task_name"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Task-specific neighbor tools (task baked into function name)
# ---------------------------------------------------------------------------

def _task_alias(task: str) -> str:
    """Return the short alias for *task*, falling back to the lowercased name."""
    return TASK_ALIASES.get(task, task.lower())


def _make_task_neighbors_tool_schema(task: str) -> Dict[str, Any]:
    """Generate a task-specific v11 neighbors schema with task baked in."""
    alias = _task_alias(task)
    return {
        "type": "function",
        "function": {
            "name": f"get_neighbors_{alias}",
            "description": (
                f"Retrieve close analogs for a molecule within the {task} prediction task. "
                "Use this to compare the query against similar compounds from the same task, "
                "inspect their labels, and optionally review selected feature groups for each analog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Query molecule, provided as a SMILES string.",
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional feature groups to compute for each retrieved analog.",
                    },
                    "include_labels": {
                        "type": "boolean",
                        "description": "Whether to include observed task labels for the retrieved analogs. Default true.",
                    },
                },
                "required": ["smiles"],
                "additionalProperties": False,
            },
        },
    }


def _make_task_neighbors_callable(task: str):
    """Return a callable that calls get_neighbors with task pre-filled."""
    alias = _task_alias(task)

    def _get_neighbors(
        smiles: str,
        feature_names: Optional[List[str]] = None,
        include_labels: bool = True,
    ) -> str:
        return get_neighbors(
            smiles, task_name=task,
            feature_names=feature_names,
            include_labels=include_labels,
        )

    _get_neighbors.__name__ = f"get_neighbors_{alias}"
    return _get_neighbors


TASK_NEIGHBOR_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    task: _make_task_neighbors_tool_schema(task) for task in TASKS
}

TASK_NEIGHBOR_CALLABLES: Dict[str, Any] = {
    f"get_neighbors_{_task_alias(task)}": _make_task_neighbors_callable(task)
    for task in TASKS
}
