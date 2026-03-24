"""
Tool 1: Molecular Profile — the "compound card".

Returns identity, drug-likeness, physicochemical properties,
complexity metrics, and electronic properties in one call.
"""

from typing import Dict, Any


def get_molecule_profile(smiles: str) -> str:
    """
    Return a comprehensive molecular profile (the "compound card").

    Includes identity, drug-likeness, key physicochemical properties,
    complexity metrics, and electronic properties (Gasteiger charges,
    EState indices, and xTB HOMO/LUMO when available).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with the molecular profile.
    """
    from . import metadata_cache
    cached = metadata_cache.lookup_row(smiles)

    def _c(prop, compute_fn):
        """Return cached value if available, else compute."""
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    import os
    from collections import Counter
    from rdkit import Chem, RDConfig
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, QED as QEDModule, ChemicalFeatures, GraphDescriptors, AllChem

    mol = _mol_from_smiles(smiles)

    # Identity
    mw = _c("MolWt", lambda: float(Descriptors.MolWt(mol)))
    heavy_atoms = int(_c("HeavyAtomCount", lambda: float(Descriptors.HeavyAtomCount(mol))))
    heteroatoms = int(_c("NumHeteroatoms", lambda: float(Lipinski.NumHeteroatoms(mol))))

    # Key physicochemical
    logp = _c("MolLogP", lambda: float(Crippen.MolLogP(mol)))
    tpsa = _c("TPSA", lambda: float(rdMolDescriptors.CalcTPSA(mol)))
    hbd = int(_c("NumHDonors", lambda: float(Lipinski.NumHDonors(mol))))
    hba = int(_c("NumHAcceptors", lambda: float(Lipinski.NumHAcceptors(mol))))
    rotatable = int(_c("NumRotatableBonds", lambda: float(Lipinski.NumRotatableBonds(mol))))
    fsp3 = _c("FractionCSP3", lambda: float(rdMolDescriptors.CalcFractionCSP3(mol)))
    mr = _c("MolMR", lambda: float(Crippen.MolMR(mol)))

    # Drug-likeness
    qed = _c("qed", lambda: float(QEDModule.qed(mol)))
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    # Complexity
    bertz = _c("BertzCT", lambda: float(Descriptors.BertzCT(mol)))
    hall_kier = _c("HallKierAlpha", lambda: float(Descriptors.HallKierAlpha(mol)))
    kappa1 = _c("Kappa1", lambda: float(GraphDescriptors.Kappa1(mol)))
    kappa2 = _c("Kappa2", lambda: float(GraphDescriptors.Kappa2(mol)))
    kappa3 = _c("Kappa3", lambda: float(GraphDescriptors.Kappa3(mol)))
    balaban_j = _c("BalabanJ", lambda: float(GraphDescriptors.BalabanJ(mol)))
    ipc = _c("Ipc", lambda: float(Descriptors.Ipc(mol)))
    amide_bonds = int(_c("NumAmideBonds", lambda: float(Lipinski.NumAmideBonds(mol))))
    stereocenters = int(_c("NumAtomStereoCenters", lambda: float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))))

    # Pharmacophore features
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    feats = factory.GetFeaturesForMol(mol)
    pharm_counts = Counter(f.GetFamily() for f in feats)

    # Electronic properties — Gasteiger charges & EState
    import numpy as np
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

    lipinski_details = []
    if mw > 500: lipinski_details.append("MW > 500")
    if logp > 5: lipinski_details.append("logP > 5")
    if hbd > 5:  lipinski_details.append("HBD > 5")
    if hba > 10: lipinski_details.append("HBA > 10")

    lines = [
        "Molecular Profile:",
        f"- Molecular weight: {mw:.2f} Da",
        f"- Heavy atoms: {heavy_atoms}, Heteroatoms: {heteroatoms}",
        f"- logP (Wildman-Crippen): {logp:.4f}",
        f"- TPSA: {tpsa:.2f} Å²",
        f"- H-bond donors: {hbd}, H-bond acceptors: {hba}",
        f"- Rotatable bonds: {rotatable}",
        f"- Fraction sp3 carbons (Fsp3): {fsp3:.4f}",
        f"- Molar refractivity: {mr:.2f}",
        f"- QED (drug-likeness, 0-1): {qed:.4f}",
        f"- Lipinski violations: {lipinski_violations}" + (f" ({', '.join(lipinski_details)})" if lipinski_details else ""),
        f"- Bertz complexity: {bertz:.2f}",
        f"- Topology: HallKierAlpha={hall_kier:.4f}, Kappa1={kappa1:.4f}, Kappa2={kappa2:.4f}, Kappa3={kappa3:.4f}, BalabanJ={balaban_j:.4f}, IPC={ipc:.4f}",
        f"- Amide bonds: {amide_bonds}",
        f"- Stereocenters: {stereocenters}",
        f"- Pharmacophore features: "
        f"{pharm_counts.get('Hydrophobe', 0)} hydrophobic, "
        f"{pharm_counts.get('Aromatic', 0)} aromatic, "
        f"{pharm_counts.get('NegIonizable', 0)} neg-ionizable, "
        f"{pharm_counts.get('PosIonizable', 0)} pos-ionizable",
        f"- Gasteiger charges: max={max_charge[2]:.4f} (atom {max_charge[0]}, {max_charge[1]}), min={min_charge[2]:.4f} (atom {min_charge[0]}, {min_charge[1]})",
        f"- Charge polarization: {charge_polarization:.4f}",
        f"- EState indices: max={max_estate:.4f} (nucleophilic proxy), min={min_estate:.4f} (electrophilic proxy)",
    ]

    # xTB quantum properties (optional — only if xtb is installed)
    try:
        from .electronic import _compute_xtb_properties
        xtb_result = _compute_xtb_properties(smiles)
        lines.append(xtb_result)
    except ImportError:
        pass
    except Exception:
        pass

    return "\n".join(lines)


def _mol_from_smiles(smiles: str):
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


# OpenAI tool schema
TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_molecule_profile",
        "description": (
            "Get a comprehensive molecular profile including identity (MW), "
            "drug-likeness (QED, Lipinski violations), key physicochemical properties "
            "(logP, TPSA, HBD, HBA, rotatable bonds, Fsp3, molar refractivity), "
            "complexity metrics (Bertz CT, stereocenters), topology indices (Hall-Kier, Kappa, Balaban J, IPC), "
            "pharmacophore feature counts (hydrophobic, aromatic, ionizable), and "
            "electronic properties (Gasteiger charges, EState indices, HOMO/LUMO gap)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."}
            },
            "required": ["smiles"],
            "additionalProperties": False,
        }
    }
}
