"""
Tool 1: Molecular Profile — the "compound card".

Returns identity, drug-likeness, physicochemical properties,
complexity metrics, and electronic properties in one call.
"""

from typing import Dict, Any


def get_molecule_profile(
    smiles: str,
    include_lipinski_violations: bool = True,
    include_electronic_summary: bool = True,
    include_quantum_properties: bool = True,
) -> str:
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
    from . import endpoints, metadata_cache
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
    mw = endpoints.mol_weight(smiles)
    heavy_atoms = int(_c("HeavyAtomCount", lambda: float(Descriptors.HeavyAtomCount(mol))))
    heteroatoms = int(_c("NumHeteroatoms", lambda: float(Lipinski.NumHeteroatoms(mol))))

    # Key physicochemical
    logp = endpoints.logp(smiles)
    tpsa = endpoints.tpsa(smiles)
    hbd = int(_c("NumHDonors", lambda: float(Lipinski.NumHDonors(mol))))
    hba = int(_c("NumHAcceptors", lambda: float(Lipinski.NumHAcceptors(mol))))
    rotatable = int(_c("NumRotatableBonds", lambda: float(Lipinski.NumRotatableBonds(mol))))
    fsp3 = _c("FractionCSP3", lambda: float(rdMolDescriptors.CalcFractionCSP3(mol)))
    mr = _c("MolMR", lambda: float(Crippen.MolMR(mol)))

    # Drug-likeness
    qed = endpoints.qed(smiles)
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    # Complexity
    bertz = _c("BertzCT", lambda: float(Descriptors.BertzCT(mol)))
    hall_kier = _c("HallKierAlpha", lambda: float(Descriptors.HallKierAlpha(mol)))
    kappa1 = _c("Kappa1", lambda: float(GraphDescriptors.Kappa1(mol)))
    kappa2 = _c("Kappa2", lambda: float(GraphDescriptors.Kappa2(mol)))
    kappa3 = _c("Kappa3", lambda: float(GraphDescriptors.Kappa3(mol)))
    balaban_j = _c("BalabanJ", lambda: float(GraphDescriptors.BalabanJ(mol)))
    amide_bonds = int(_c("NumAmideBonds", lambda: float(Lipinski.NumAmideBonds(mol))))
    stereocenters = int(_c("NumAtomStereoCenters", lambda: float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))))

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
        f"- Molecular weight: {mw:.2f} Da",
        f"- Heavy atoms: {heavy_atoms}, Heteroatoms: {heteroatoms}",
        f"- logP (Wildman-Crippen): {logp:.2f}",
        f"- TPSA: {tpsa:.2f} Å²",
        f"- H-bond donors: {hbd}, H-bond acceptors: {hba}",
        f"- Rotatable bonds: {rotatable}",
        f"- Fraction sp3 carbons (Fsp3): {fsp3:.2f}",
        f"- Molar refractivity: {mr:.2f}",
        f"- QED (drug-likeness, 0-1): {qed:.2f}",
        f"- Bertz complexity: {bertz:.2f}",
        f"- Topology: HallKierAlpha={hall_kier:.2f}, Kappa1={kappa1:.2f}, Kappa2={kappa2:.2f}, Kappa3={kappa3:.2f}, BalabanJ={balaban_j:.2f}",
        f"- Amide bonds: {amide_bonds}",
        f"- Stereocenters: {stereocenters}",
        f"- Pharmacophore features: "
        f"{pharm_counts.get('Hydrophobe', 0)} hydrophobic, "
        f"{pharm_counts.get('Aromatic', 0)} aromatic, "
        f"{pharm_counts.get('NegIonizable', 0)} neg-ionizable, "
        f"{pharm_counts.get('PosIonizable', 0)} pos-ionizable",
    ]

    if include_lipinski_violations:
        lines.insert(
            11,
            f"- Lipinski violations: {lipinski_violations}" + (f" ({', '.join(lipinski_details)})" if lipinski_details else ""),
        )

    if include_electronic_summary:
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

        lines.extend([
            f"- Gasteiger charges: max={max_charge[2]:.2f} (atom {max_charge[0]}, {max_charge[1]}), min={min_charge[2]:.2f} (atom {min_charge[0]}, {min_charge[1]})",
            f"- Charge polarization: {charge_polarization:.2f}",
            f"- EState indices: max={max_estate:.2f} (nucleophilic proxy), min={min_estate:.2f} (electrophilic proxy)",
        ])

    if include_quantum_properties:
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
            "Compute physicochemical profile: MW, logP, TPSA, HBD/HBA, rotatable bonds, Fsp3, QED, "
            "Lipinski violations, complexity, topology indices, pharmacophore counts, and electronic properties."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"}
            },
            "required": ["smiles"],
        }
    }
}
