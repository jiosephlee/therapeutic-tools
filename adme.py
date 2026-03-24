"""
Tool 4: ADME Properties — ionization, partitioning, solubility.

These properties are causally linked: pKa → ionization → logD → permeability.
TPSA and surface area live in get_molecule_profile / get_3d_properties.
"""

from typing import Dict, Any


def assess_adme_properties(smiles: str, ph: float = 7.4) -> str:
    """
    Assess ADME-related properties: ionization, partitioning, solubility, and surface area.

    These properties are causally linked: pKa → ionization → logD → permeability.

    Args:
        smiles: SMILES string of the molecule.
        ph: Target pH for ionization/logD estimation (default: 7.4).

    Returns:
        Multi-line formatted string with ADME property assessment.
    """
    from .legacy_tools.RDKit_tools import get_esol
    from . import metadata_cache

    cached = metadata_cache.lookup_row(smiles)
    sections = []

    # pKa prediction — use cache if available, else compute
    pka_data = _get_pka_data(smiles, cached)
    lines = ["pKa Prediction:"]

    acid_sites = pka_data.get("acid_sites")  # {atom_map: pKa} or None
    base_sites = pka_data.get("base_sites")

    if acid_sites:
        site_strs = [f"atom {atom} pKa = {pka:.2f}" for atom, pka in sorted(acid_sites.items(), key=lambda x: x[1])]
        lines.append(f"  Acidic sites ({len(acid_sites)}): {', '.join(site_strs)}")
    elif pka_data["num_acidic_sites"] > 0:
        # Cache fallback — no per-site detail
        lines.append(f"  Acidic sites ({pka_data['num_acidic_sites']}): most acidic pKa = {pka_data['most_acidic_pka']:.2f}")
    else:
        lines.append("  No acidic sites.")

    if base_sites:
        site_strs = [f"atom {atom} pKa = {pka:.2f}" for atom, pka in sorted(base_sites.items(), key=lambda x: -x[1])]
        lines.append(f"  Basic sites ({len(base_sites)}): {', '.join(site_strs)}")
    elif pka_data["num_basic_sites"] > 0:
        lines.append(f"  Basic sites ({pka_data['num_basic_sites']}): most basic pKa = {pka_data['most_basic_pka']:.2f}")
    else:
        lines.append("  No basic sites.")

    sections.append("\n".join(lines))

    # Ionization classification (compact format)
    try:
        ionization_result = _compact_ionization(smiles, ph=ph)
        sections.append(ionization_result)
    except Exception as e:
        sections.append(f"\nIonization Classification: Error - {e}")

    # LogD estimation — computed from pKa data, no redundant recomputation
    logd = _estimate_logd_from_pka(smiles, ph, pka_data, cached)
    sections.append(f"\nLogD at pH {ph}: {logd:.2f}")

    # Solubility — use ML prediction from cache if available, else ESOL heuristic
    if cached and "minimol_solubility_log_mol_L" in cached:
        sections.append(f"Solubility: logS = {cached['minimol_solubility_log_mol_L']:.2f} log(mol/L)")
    else:
        try:
            esol_result = get_esol(smiles)
            sections.append(f"Solubility (ESOL heuristic): {esol_result}")
        except Exception as e:
            sections.append(f"Solubility: Error - {e}")

    return "\n".join(sections)


def _compact_ionization(smiles: str, ph: float = 7.4) -> str:
    """Compact ionization state classification. 3 lines for simple cases,
    expanded for ampholytes/zwitterions."""
    from rdkit import Chem
    from collections import Counter
    from .legacy_tools.RDKit_tools import protonate_smiles

    mol = Chem.MolFromSmiles(smiles)
    canon_smi = Chem.MolToSmiles(mol, canonical=True)

    try:
        variants = protonate_smiles(canon_smi, ph_min=ph, ph_max=ph, precision=0.5)
    except Exception:
        variants = []

    variant_data = []
    for v_smi in variants:
        m = Chem.MolFromSmiles(v_smi)
        if m is None:
            continue
        pos = neg = 0
        for atom in m.GetAtoms():
            fc = atom.GetFormalCharge()
            if fc > 0:
                pos += fc
            elif fc < 0:
                neg += abs(fc)
        net = pos - neg
        if pos > 0 and neg > 0:
            cc = "zwitterion"
        elif pos > 0:
            cc = "base"
        elif neg > 0:
            cc = "acid"
        else:
            cc = "neutral"
        variant_data.append({"smiles": v_smi, "net_charge": net, "charge_class": cc})

    if not variant_data:
        variant_data = [{"smiles": canon_smi, "net_charge": 0, "charge_class": "neutral"}]

    net_charges = sorted(set(v["net_charge"] for v in variant_data))
    charge_class_counts = Counter(v["charge_class"] for v in variant_data)
    net_charge_counts = Counter(v["net_charge"] for v in variant_data)
    mode_charge = max(net_charge_counts.keys(), key=lambda c: (net_charge_counts[c], -abs(c)))
    representative = next(v for v in variant_data if v["net_charge"] == mode_charge)
    is_ambiguous = len(net_charges) > 1

    lines = [
        f"Ionization at pH {ph}: {representative['charge_class']}, charge {representative['net_charge']}",
        f"- Dominant form: {representative['smiles']}",
        f"- Ambiguous (pKa near pH): {'Yes' if is_ambiguous else 'No'}",
    ]

    # Expanded detail only for multi-class cases (ampholytes/zwitterions)
    if len(charge_class_counts) > 1:
        dist = ", ".join(f"{k}: {v}" for k, v in charge_class_counts.items())
        lines.append(f"- Charge class distribution: {dist}")

    return "\n".join(lines)


def _molgpka_to_heavy_atom_idx(sites: dict, smiles: str) -> dict:
    """Convert MolGpKa atom map numbers (1-based, H-added mol) to 0-based heavy atom indices.

    MolGpKa assigns atom_map = idx+1 on AddHs(mol). If the site is on a hydrogen,
    we report the heavy atom it's attached to instead. Returns {heavy_atom_idx: pKa}."""
    from rdkit import Chem
    if not sites:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return sites
    mol_h = Chem.AddHs(mol)
    result = {}
    for map_num, pka in sites.items():
        h_idx = int(map_num) - 1  # 1-based -> 0-based in H-added mol
        if h_idx >= mol_h.GetNumAtoms():
            continue
        atom = mol_h.GetAtomWithIdx(h_idx)
        if atom.GetAtomicNum() == 1:  # hydrogen — find parent heavy atom
            neighbors = atom.GetNeighbors()
            if neighbors:
                heavy_idx = neighbors[0].GetIdx()
            else:
                continue
        else:
            heavy_idx = h_idx
        # Only include if it's a valid heavy atom index in the original mol
        if heavy_idx < mol.GetNumAtoms():
            result[heavy_idx] = float(pka)
    return result if result else {}


def _parse_pka_sites_from_cache(smiles: str, column: str):
    """Load per-site pKa JSON from metadata cache. Returns {atom: pKa} or None."""
    try:
        import json
        import pandas as pd
        from . import metadata_cache
        meta = metadata_cache._load_metadata()
        if meta is None or smiles not in meta.index:
            return None
        row = meta.loc[smiles]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if column not in row.index or pd.isna(row[column]):
            return None
        raw = json.loads(row[column])
        return {int(k): float(v) for k, v in raw.items()} if raw else None
    except Exception:
        return None


def _get_pka_data(smiles: str, cached) -> dict:
    """Extract pKa data from cache or compute via MolGpKa. Returns dict with keys:
    most_acidic_pka, most_basic_pka, num_acidic_sites, num_basic_sites,
    acid_sites (dict atom->pKa or None), base_sites (dict atom->pKa or None)."""
    if cached and "most_acidic_pka" in cached and "num_acidic_sites" in cached:
        # Try to load per-site JSON from cache (stored as MolGpKa map numbers)
        acid_sites_raw = _parse_pka_sites_from_cache(smiles, "acid_sites_json")
        base_sites_raw = _parse_pka_sites_from_cache(smiles, "base_sites_json")
        # Convert to 0-based heavy atom indices
        acid_sites = _molgpka_to_heavy_atom_idx(acid_sites_raw, smiles) if acid_sites_raw else None
        base_sites = _molgpka_to_heavy_atom_idx(base_sites_raw, smiles) if base_sites_raw else None
        return {
            "most_acidic_pka": cached.get("most_acidic_pka"),
            "most_basic_pka": cached.get("most_basic_pka"),
            "num_acidic_sites": int(cached.get("num_acidic_sites", 0)),
            "num_basic_sites": int(cached.get("num_basic_sites", 0)),
            "acid_sites": acid_sites or None,
            "base_sites": base_sites or None,
        }
    # Compute live
    from .legacy_tools.pka_related_tools import _mol_from_smiles, _get_pka_predictor
    mol = _mol_from_smiles(smiles)
    predictor = _get_pka_predictor()
    prediction = predictor.predict(mol)
    base_sites_raw = prediction.base_sites_1  # {atom_map_number: pKa}
    acid_sites_raw = prediction.acid_sites_1
    # Convert MolGpKa 1-based map numbers to 0-based heavy atom indices
    acid_sites = _molgpka_to_heavy_atom_idx(dict(acid_sites_raw), smiles) if acid_sites_raw else None
    base_sites = _molgpka_to_heavy_atom_idx(dict(base_sites_raw), smiles) if base_sites_raw else None
    return {
        "most_acidic_pka": min(acid_sites_raw.values()) if acid_sites_raw else None,
        "most_basic_pka": max(base_sites_raw.values()) if base_sites_raw else None,
        "num_acidic_sites": len(acid_sites_raw),
        "num_basic_sites": len(base_sites_raw),
        "acid_sites": acid_sites,
        "base_sites": base_sites,
    }


def _estimate_logd_from_pka(smiles: str, ph: float, pka_data: dict, cached) -> float:
    """Estimate logD using Henderson-Hasselbalch from pKa data already computed."""
    import math
    from rdkit.Chem import Crippen
    from rdkit import Chem

    if cached and "logD_74" in cached and abs(ph - 7.4) < 0.01:
        return float(cached["logD_74"])

    mol = Chem.MolFromSmiles(smiles)
    logp = float(Crippen.MolLogP(mol))

    f_neutral = 1.0
    if pka_data["most_basic_pka"] is not None:
        f_neutral *= 1.0 / (1.0 + 10.0 ** (pka_data["most_basic_pka"] - ph))
    if pka_data["most_acidic_pka"] is not None:
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - pka_data["most_acidic_pka"]))

    f_neutral = min(1.0, max(1e-12, f_neutral))
    return logp + math.log10(f_neutral)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "assess_adme_properties",
        "description": (
            "Assess ADME properties: pKa prediction (acidic/basic sites), "
            "ionization state classification at target pH, logD estimation, "
            "and aqueous solubility. "
            "These are causally linked: pKa → ionization → logD → permeability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."},
                "ph": {"type": "number", "description": "Target pH for ionization/logD estimation (default: 7.4)."}
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
