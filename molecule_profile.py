"""
Tool 1: Molecular Profile — the "compound card".

Returns identity, drug-likeness, physicochemical properties,
charge information, and complexity metrics in one call.
"""

from typing import Dict, Any


def get_molecule_profile(smiles: str) -> str:
    """
    Return a comprehensive molecular profile (the "compound card").

    Includes identity, drug-likeness, key physicochemical properties,
    charge information, and complexity metrics.

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
    from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, QED as QEDModule, ChemicalFeatures

    mol = _mol_from_smiles(smiles)

    # Identity
    mw = _c("MolWt", lambda: float(Descriptors.MolWt(mol)))
    exact_mw = _c("ExactMolWt", lambda: float(Descriptors.ExactMolWt(mol)))
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

    # Charge
    formal_charge = int(_c("FormalCharge", lambda: float(Chem.GetFormalCharge(mol))))
    n_pos = int(_c("NumPositiveCharges", lambda: float(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0))))
    n_neg = int(_c("NumNegativeCharges", lambda: float(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0))))

    # Complexity
    bertz = _c("BertzCT", lambda: float(Descriptors.BertzCT(mol)))
    amide_bonds = int(_c("NumAmideBonds", lambda: float(Lipinski.NumAmideBonds(mol))))
    stereocenters = int(_c("NumAtomStereoCenters", lambda: float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))))
    unspecified_stereo = int(_c("NumUnspecifiedAtomStereoCenters", lambda: float(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol))))

    # Pharmacophore features
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    feats = factory.GetFeaturesForMol(mol)
    pharm_counts = Counter(f.GetFamily() for f in feats)

    lipinski_details = []
    if mw > 500: lipinski_details.append("MW > 500")
    if logp > 5: lipinski_details.append("logP > 5")
    if hbd > 5:  lipinski_details.append("HBD > 5")
    if hba > 10: lipinski_details.append("HBA > 10")

    lines = [
        "Molecular Profile:",
        f"- Molecular weight: {mw:.2f} Da (exact: {exact_mw:.4f})",
        f"- Heavy atoms: {heavy_atoms}, Heteroatoms: {heteroatoms}",
        f"- logP (Wildman-Crippen): {logp:.4f}",
        f"- TPSA: {tpsa:.2f} Å²",
        f"- H-bond donors: {hbd}, H-bond acceptors: {hba}",
        f"- Rotatable bonds: {rotatable}",
        f"- Fraction sp3 carbons (Fsp3): {fsp3:.4f}",
        f"- Molar refractivity: {mr:.2f}",
        f"- QED (drug-likeness, 0-1): {qed:.4f}",
        f"- Lipinski violations: {lipinski_violations}" + (f" ({', '.join(lipinski_details)})" if lipinski_details else ""),
        f"- Net formal charge: {formal_charge} (cationic centers: {n_pos}, anionic centers: {n_neg})",
        f"- Bertz complexity: {bertz:.2f}",
        f"- Amide bonds: {amide_bonds}",
        f"- Stereocenters: {stereocenters} ({unspecified_stereo} unspecified)",
        f"- Pharmacophore features: "
        f"{pharm_counts.get('Hydrophobe', 0)} hydrophobic, "
        f"{pharm_counts.get('LumpedHydrophobe', 0)} lumped-hydrophobic, "
        f"{pharm_counts.get('Aromatic', 0)} aromatic, "
        f"{pharm_counts.get('NegIonizable', 0)} neg-ionizable, "
        f"{pharm_counts.get('PosIonizable', 0)} pos-ionizable, "
        f"{pharm_counts.get('ZnBinder', 0)} Zn-binder",
    ]
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
            "Get a comprehensive molecular profile including identity (MW, formula), "
            "drug-likeness (QED, Lipinski violations), key physicochemical properties "
            "(logP, TPSA, HBD, HBA, rotatable bonds, Fsp3, molar refractivity), "
            "charge information, complexity metrics (Bertz CT, stereocenters), and "
            "pharmacophore feature counts (hydrophobic, aromatic, ionizable, Zn-binder)."
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
