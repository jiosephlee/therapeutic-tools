"""
Expansion Tool: Metabolic Site Prediction — CYP450 soft spots.

Predicts which atoms/bonds on a molecule are most likely to be oxidized by
cytochrome P450 enzymes.

Backends (tried in order by "auto"):
  1. FAME3R   — generic Phase 1 & 2 SoM (pip install fame3r, Python >= 3.10)
  2. RDKit    — SMARTS-based heuristic, zero extra deps

Why this tool matters:
  - CYP inhibition tasks (e.g. CYP2C19): SoM shows WHERE a CYP isoform
    attacks -> model reasons about active-site binding vs. being metabolized.
  - DILI: SoM -> identify reactive metabolite formation sites (epoxides,
    quinone methides, etc.) that trigger GSH depletion -> hepatotoxicity.
  - AMES: Metabolic activation can convert pro-mutagens into DNA-reactive species.
"""

from typing import Dict, Any, List, Tuple


def predict_metabolism_sites(smiles: str, engine: str = "auto") -> str:
    """
    Predict sites of CYP450 metabolism (soft spots) on a molecule.

    Returns atom-level predictions showing which sites are most
    susceptible to oxidative metabolism.

    Args:
        smiles: SMILES string of the molecule.
        engine: Backend to use. "auto" tries ATTNSOM -> FAME3R -> RDKit in order.
                Options: "attnsom", "fame3r", "rdkit", "auto".

    Returns:
        Multi-line formatted string with metabolism site predictions.
    """
    if engine == "auto":
        try:
            return _predict_attnsom(smiles)
        except (ImportError, FileNotFoundError):
            pass
        try:
            return _predict_fame3r(smiles)
        except ImportError:
            pass
        return _predict_rdkit_heuristic(smiles)
    elif engine == "attnsom":
        return _predict_attnsom(smiles)
    elif engine == "fame3r":
        return _predict_fame3r(smiles)
    elif engine == "rdkit":
        return _predict_rdkit_heuristic(smiles)
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'auto', 'attnsom', 'fame3r', or 'rdkit'.")


_ATTNSOM_CACHE = None


def _load_attnsom_cache():
    """Lazy-load the precomputed ATTNSOM JSONL cache into a dict keyed by canonical SMILES."""
    global _ATTNSOM_CACHE
    if _ATTNSOM_CACHE is not None:
        return _ATTNSOM_CACHE
    import os, json
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "attnsom_cache.jsonl")
    _ATTNSOM_CACHE = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    _ATTNSOM_CACHE[entry["smiles"]] = entry["result"]
                except Exception:
                    pass
    return _ATTNSOM_CACHE


def _predict_attnsom(smiles: str) -> str:
    """Predict metabolism sites using ATTNSOM (GNN + cross-isoform attention).

    Checks the precomputed disk cache first; falls back to live inference.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canon_smi = Chem.MolToSmiles(mol)

    # Check disk cache
    cache = _load_attnsom_cache()
    if canon_smi in cache:
        return cache[canon_smi]

    # Cache miss — run live inference
    import os
    from .ATTNSOM.inference import ATTNSOMPredictor, format_prediction

    global _ATTNSOM_PREDICTOR
    if "_ATTNSOM_PREDICTOR" not in globals() or _ATTNSOM_PREDICTOR is None:
        ckpt_path = os.environ.get("ATTNSOM_CHECKPOINT")
        if not ckpt_path:
            candidates = [
                "/vast/projects/myatskar/design-documents/hf_home/attnsom_results/attnsom_checkpoint.pt",
                os.path.join(os.path.dirname(__file__), "ATTNSOM", "results", "attnsom_checkpoint.pt"),
            ]
            ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ATTNSOM checkpoint not found at {ckpt_path}")
        _ATTNSOM_PREDICTOR = ATTNSOMPredictor(ckpt_path)

    result = _ATTNSOM_PREDICTOR.predict(canon_smi)
    return format_prediction(result)


_ATTNSOM_PREDICTOR = None


def _predict_fame3r(smiles: str) -> str:
    """Predict metabolism sites using FAME3R (CDPKit + trained RandomForest).

    FAME3R does NOT ship a pretrained model. You must first train one:
        fame3r train -i training_data.sdf -o /path/to/models/

    Then set FAME3R_MODEL_DIR env var to the directory containing the trained
    model files (e.g. all/random_forest_classifier.joblib).

    Falls back: set FAME3R_MODEL_DIR to a directory with subdirectory 'all/'
    containing 'random_forest_classifier.joblib'.
    """
    import os
    import numpy as np

    model_dir = os.environ.get("FAME3R_MODEL_DIR")
    if not model_dir:
        raise ImportError("FAME3R_MODEL_DIR not set")

    # Look for the trained classifier
    clf_path = os.path.join(model_dir, "all", "random_forest_classifier.joblib")
    if not os.path.exists(clf_path):
        # Also try direct path
        clf_path = os.path.join(model_dir, "random_forest_classifier.joblib")
        if not os.path.exists(clf_path):
            raise ImportError(f"No trained FAME3R model found at {model_dir}")

    import joblib
    from CDPL.Chem import parseSMILES
    from fame3r import FAME3RVectorizer
    from sklearn.pipeline import make_pipeline

    # Parse SMILES with CDPKit
    cdpkit_mol = parseSMILES(smiles)

    # Build atom array (CDPKit atoms need special handling for numpy)
    atoms = list(cdpkit_mol.atoms)
    n_atoms = len(atoms)
    atom_array = np.empty((n_atoms, 1), dtype=object)
    atom_array[:, 0] = atoms

    # Build prediction pipeline: vectorizer + trained classifier
    classifier = joblib.load(clf_path)
    pipeline = make_pipeline(
        FAME3RVectorizer(input="cdpkit").fit(),
        classifier,
    )

    # Get per-atom SoM probabilities
    probs = pipeline.predict_proba(atom_array)[:, 1]

    # Also get RDKit mol for atom symbol lookup
    from rdkit import Chem
    rdkit_mol = Chem.MolFromSmiles(smiles)

    # Rank by probability
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    significant = [(idx, p) for idx, p in ranked if p > 0.1] or ranked[:5]

    lines = ["CYP450 Sites of Metabolism (FAME3R — Phase 1 & 2):", ""]
    for rank, (atom_idx, prob) in enumerate(significant[:10], 1):
        try:
            if rdkit_mol is not None:
                atom = rdkit_mol.GetAtomWithIdx(atom_idx)
                mech = _infer_mechanism(rdkit_mol, atom_idx)
                lines.append(f"  {rank}. Atom {atom_idx} ({atom.GetSymbol()}), p = {prob:.3f} — {mech}")
            else:
                lines.append(f"  {rank}. Atom {atom_idx}, p = {prob:.3f}")
        except Exception:
            lines.append(f"  {rank}. Atom {atom_idx}, p = {prob:.3f}")

    n_high = len([p for _, p in ranked if p > 0.3])
    lines += ["", f"Summary: {n_high} high-probability sites (p > 0.3)"]
    return "\n".join(lines)


def _infer_mechanism(mol, atom_idx: int) -> str:
    """Infer metabolic mechanism from atom chemical environment."""
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    is_aromatic = atom.GetIsAromatic()
    neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]

    if symbol == "C":
        if is_aromatic:
            return "aromatic hydroxylation"
        if any(n.GetIsAromatic() for n in neighbors):
            return "benzylic hydroxylation (CYP3A4, CYP2C9)"
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                return "allylic/vinylic oxidation"
        return "aliphatic hydroxylation"
    elif symbol == "N":
        c_neighbors = [n for n in neighbors if n.GetSymbol() == "C" and not n.GetIsAromatic()]
        if c_neighbors:
            return "N-dealkylation (CYP3A4, CYP2D6)"
        return "N-oxidation"
    elif symbol == "O":
        return "O-dealkylation (CYP2D6)"
    elif symbol == "S":
        return "S-oxidation -> sulfoxide/sulfone"
    return "oxidation"


def _predict_rdkit_heuristic(smiles: str) -> str:
    """SMARTS-based heuristic (zero extra deps)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    PATTERNS = [
        ("Benzylic C-H",     "[CH2,CH1;$([CH2,CH1]c)]",         "Benzylic hydroxylation (CYP3A4, CYP2C9)"),
        ("Allylic C-H",      "[CH2,CH1;$([CH2,CH1]C=C)]",       "Allylic hydroxylation"),
        ("N-dealkylation",   "[#7;$([#7]([CH3,CH2]))]",          "N-dealkylation (CYP3A4, CYP2D6)"),
        ("O-dealkylation",   "[OX2;$([OX2][CH3,CH2])]",          "O-dealkylation (CYP2D6)"),
        ("S-oxidation",      "[#16X2]",                           "S-oxidation -> sulfoxide/sulfone"),
        ("Aromatic-OH",      "[cH;$([cH]1cc([!H])ccc1)]",        "Aromatic hydroxylation (para)"),
        ("N-oxidation",      "[nX2,NX3;!$([NH2])!$([NH]C=O)]",   "N-oxidation"),
        ("Aldehyde",         "[CH]=O",                             "Aldehyde oxidation"),
        ("w-oxidation",      "[CH3;!$([CH3]a)]",                  "Terminal methyl hydroxylation"),
        ("Epoxidation",      "[C]=[C;$([C](~[!H]))]",             "Epoxidation — reactive metabolite risk (DILI)"),
    ]

    hits: List[Tuple[str, str, List[int]]] = []
    for name, pat, desc in PATTERNS:
        q = Chem.MolFromSmarts(pat)
        if q is None:
            continue
        matches = mol.GetSubstructMatches(q)
        if matches:
            ids = sorted(set(i for m in matches for i in m))
            hits.append((name, desc, ids))

    lines = [
        "CYP450 Sites of Metabolism (RDKit SMARTS heuristic):",
        "",
    ]
    if hits:
        for i, (name, desc, ids) in enumerate(hits, 1):
            atoms_str = ", ".join(
                f"atom {idx}({mol.GetAtomWithIdx(idx).GetSymbol()})" for idx in ids[:5]
            )
            if len(ids) > 5:
                atoms_str += f" +{len(ids)-5} more"
            lines.append(f"{i}. {name}: {atoms_str}")
            lines.append(f"   -> {desc}")
        has_epox = any(h[0] == "Epoxidation" for h in hits)
        lines += [
            "",
            f"Summary: {sum(len(h[2]) for h in hits)} vulnerable sites detected",
            f"Reactive metabolite risk: {'Yes (epoxidation site)' if has_epox else 'Low'}",
        ]
    else:
        lines += ["No common metabolic soft spots detected", "Low CYP-mediated clearance expected"]

    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "predict_metabolism_sites",
        "description": (
            "Predict which atoms in a molecule are most likely to be oxidized by "
            "cytochrome P450 enzymes. Uses ATTNSOM (GNN with cross-isoform attention, "
            "9 CYP isoforms), FAME3R, or RDKit SMARTS heuristics as fallback. "
            "Use for: (1) CYP inhibition tasks — SoM reasoning "
            "about enzyme-substrate interactions; (2) DILI — identifying reactive metabolite "
            "formation sites; (3) AMES — metabolic activation of pro-mutagens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."},
                "engine": {
                    "type": "string",
                    "description": "Prediction backend: 'auto' (default), 'attnsom', 'fame3r', or 'rdkit'.",
                    "enum": ["auto", "attnsom", "fame3r", "rdkit"]
                }
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
