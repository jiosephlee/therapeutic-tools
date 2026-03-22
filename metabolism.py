"""
Expansion Tool: Metabolic Site Prediction & Metabolite Prediction.

Two complementary predictions:
  1. Sites of Metabolism (SoM): ATTNSOM (isoform-specific CYP Phase 1) or
     RDKit SMARTS heuristic fallback.
  2. Metabolite Structures: SyGMa (rule-based Phase 1 + Phase 2 metabolite
     prediction with scored pathways).

Why this tool matters:
  - CYP inhibition tasks (e.g. CYP2C19): SoM shows WHERE a CYP isoform
    attacks -> model reasons about active-site binding vs. being metabolized.
  - DILI: SoM -> identify reactive metabolite formation sites (epoxides,
    quinone methides, etc.) that trigger GSH depletion -> hepatotoxicity.
    SyGMa shows Phase 2 conjugation (glucuronidation, GSH) that detoxifies.
  - AMES: Metabolic activation can convert pro-mutagens into DNA-reactive species.
    SyGMa predicts the actual metabolite structures formed.
"""

from typing import Dict, Any, List, Tuple, Optional

ALL_ISOFORMS = ["1A2", "2A6", "2B6", "2C8", "2C9", "2C19", "2D6", "2E1", "3A4"]


def predict_metabolism_sites(
    smiles: str,
    isoforms: Optional[List[str]] = None,
) -> str:
    """
    Predict sites of CYP450 metabolism and likely metabolite structures.

    Returns:
      1. CYP SoM predictions (ATTNSOM isoform-specific, or RDKit heuristic fallback)
      2. Predicted metabolites with pathways and scores (SyGMa Phase 1 + Phase 2)

    Args:
        smiles: SMILES string of the molecule.
        isoforms: List of CYP isoforms to predict (e.g. ["2C9", "3A4"]).
                  Defaults to all 9 isoforms.

    Returns:
        Multi-line formatted string with metabolism predictions.
    """
    if isoforms is not None:
        # Validate isoform names
        invalid = [i for i in isoforms if i not in ALL_ISOFORMS]
        if invalid:
            return (
                f"Error: Unknown isoform(s): {', '.join(invalid)}. "
                f"Valid isoforms: {', '.join(ALL_ISOFORMS)}"
            )

    sections = []

    # SoM prediction: ATTNSOM -> RDKit fallback
    try:
        sections.append(_predict_attnsom(smiles, isoforms=isoforms))
    except (ImportError, FileNotFoundError):
        sections.append(_predict_rdkit_heuristic(smiles))

    # Metabolite prediction: SyGMa Phase 1 + Phase 2
    try:
        sections.append(_predict_sygma(smiles))
    except Exception:
        pass  # SyGMa is optional

    return "\n\n".join(sections)


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


def _predict_attnsom(smiles: str, isoforms: Optional[List[str]] = None) -> str:
    """Predict metabolism sites using ATTNSOM (GNN + cross-isoform attention).

    Checks the precomputed disk cache first; falls back to live inference.
    """
    from rdkit import Chem
    from .ATTNSOM.inference import ATTNSOMPredictor, format_prediction
    import json as _json

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canon_smi = Chem.MolToSmiles(mol)

    # Check disk cache — stored as pre-formatted string, need to re-parse
    # for isoform filtering
    cache = _load_attnsom_cache()
    if canon_smi in cache:
        cached_str = cache[canon_smi]
        if isoforms is None:
            return cached_str
        # Filter cached output by isoforms
        return _filter_attnsom_output(cached_str, isoforms)

    # Cache miss — run live inference
    import os

    global _ATTNSOM_PREDICTOR
    if "_ATTNSOM_PREDICTOR" not in globals() or _ATTNSOM_PREDICTOR is None:
        ckpt_path = os.environ.get("ATTNSOM_CHECKPOINT")
        if not ckpt_path:
            candidates = [
                "/vast/projects/myatskar/design-documents/hf_home/attnsom_results_v2/attnsom_checkpoint.pt",
                "/vast/projects/myatskar/design-documents/hf_home/attnsom_results/attnsom_checkpoint.pt",
                os.path.join(os.path.dirname(__file__), "ATTNSOM", "results", "attnsom_checkpoint.pt"),
            ]
            ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ATTNSOM checkpoint not found at {ckpt_path}")
        _ATTNSOM_PREDICTOR = ATTNSOMPredictor(ckpt_path)

    result = _ATTNSOM_PREDICTOR.predict(canon_smi)
    if isoforms is not None:
        result["predictions"] = [
            p for p in result["predictions"] if p["cyp"] in isoforms
        ]
    return format_prediction(result)


_ATTNSOM_PREDICTOR = None


def _filter_attnsom_output(text: str, isoforms: List[str]) -> str:
    """Filter pre-formatted ATTNSOM output to only include specified isoforms."""
    lines = text.split("\n")
    filtered = []
    include_block = True
    som_count = 0

    for line in lines:
        # Detect isoform blocks: "  CYP2C9 (model F1=..."
        if line.strip().startswith("CYP") and "(model" in line:
            cyp_name = line.strip().split("(")[0].strip().replace("CYP", "")
            include_block = cyp_name in isoforms
            if include_block:
                filtered.append(line)
            continue

        if line.strip().startswith("Summary:"):
            # Recompute summary for filtered isoforms
            filtered.append("")
            filtered.append(f"Summary: {som_count} total SoM sites across {len(isoforms)} CYP isoform(s)")
            continue

        if include_block:
            filtered.append(line)
            # Count SoM sites
            if "*SoM*" in line:
                som_count += 1

    return "\n".join(filtered)


def _predict_sygma(smiles: str, max_metabolites: int = 10) -> str:
    """Predict metabolite structures using SyGMa (Phase 1 + Phase 2 rules).

    SyGMa applies reaction SMARTS rules derived from literature to enumerate
    likely metabolites with empirical probability scores. Phase 2 covers
    glucuronidation, sulfation, GSH conjugation, acetylation, glycination.
    """
    import sygma
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    sc = sygma.Scenario([
        [sygma.ruleset['phase1'], 1],
        [sygma.ruleset['phase2'], 1],
    ])
    tree = sc.run(mol)
    tree.calc_scores()

    results = tree.to_list()
    # Skip parent molecule
    metabolites = [r for r in results if r['SyGMa_pathway'].strip() != 'parent;']

    lines = ["Predicted Metabolites (SyGMa — Phase 1 + Phase 2):", ""]
    if not metabolites:
        lines.append("No metabolites predicted.")
        return "\n".join(lines)

    for i, r in enumerate(metabolites[:max_metabolites], 1):
        met_smi = Chem.MolToSmiles(r['SyGMa_metabolite'])
        pathway = r['SyGMa_pathway'].strip().rstrip('; \n').replace('; \n', ' -> ')
        score = r['SyGMa_score']
        lines.append(f"  {i}. {met_smi}")
        lines.append(f"     Pathway: {pathway} (score={score:.4f})")

    # Classify Phase 1 vs Phase 2
    phase2_keywords = {'glucuronid', 'sulph', 'sulfat', 'GSH', 'acetyl', 'glycin',
                       'methylat', 'glutathion'}
    n_phase2 = sum(1 for r in metabolites[:max_metabolites]
                   if any(kw in r['SyGMa_pathway'].lower() for kw in phase2_keywords))
    n_phase1 = min(len(metabolites), max_metabolites) - n_phase2

    lines.append("")
    lines.append(f"Summary: {len(metabolites)} metabolites predicted "
                 f"(showing top {min(len(metabolites), max_metabolites)}: "
                 f"{n_phase1} Phase 1, {n_phase2} Phase 2)")
    return "\n".join(lines)


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
            "Predict CYP450 sites of metabolism and likely metabolite structures. "
            "Returns: (1) ATTNSOM isoform-specific SoM predictions showing which atoms "
            "each CYP isoform is most likely to oxidize; (2) SyGMa metabolite predictions "
            "with SMILES structures and pathways for both Phase 1 (oxidation, reduction, "
            "hydrolysis) and Phase 2 (glucuronidation, sulfation, GSH conjugation, "
            "acetylation, glycination). "
            "Available CYP isoforms: 1A2, 2A6, 2B6, 2C8, 2C9, 2C19, 2D6, 2E1, 3A4. "
            "Use for: (1) CYP inhibition/substrate tasks — SoM + metabolite reasoning; "
            "(2) DILI — reactive metabolites vs. detoxifying conjugation; "
            "(3) AMES — metabolic activation of pro-mutagens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."},
                "isoforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CYP isoforms to predict (e.g. ['2C9', '3A4']). "
                        "Defaults to all 9 isoforms if omitted."
                    ),
                },
            },
            "required": ["smiles", "isoforms"],
            "additionalProperties": False,
        }
    }
}
