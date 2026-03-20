"""
Expansion Tool: Electronic Properties — HOMO/LUMO, dipole, electrophilicity via xtb.

Uses GFN2-xTB semi-empirical quantum chemistry (Python API) for electronic
property calculation.  Always includes Gasteiger charges and EState indices.
"""

from typing import Dict, Any
import numpy as np


def get_electronic_properties(smiles: str) -> str:
    """
    Compute electronic properties of a molecule.

    Always reports Gasteiger partial charges (charge polarization) and
    EState indices.  When xtb is available, also reports HOMO/LUMO
    energies, dipole moment, and electrophilicity index.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with electronic property analysis.
    """
    from . import metadata_cache
    cached = metadata_cache.lookup_row(smiles)

    sections = []

    # --- Gasteiger charges & EState (always shown) ---
    sections.append(_compute_charge_estate(smiles, cached))

    # --- xTB quantum properties (if available) ---
    try:
        xtb_result = _compute_xtb_properties(smiles)
        sections.append("")
        sections.append(xtb_result)
    except ImportError:
        pass  # xtb not installed — skip silently
    except Exception as e:
        sections.append(f"\nxTB calculation failed: {e}")

    return "\n".join(sections)


def _compute_charge_estate(smiles: str, cached=None) -> str:
    """Gasteiger charges and EState indices."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        try:
            q = float(atom.GetDoubleProp("_GasteigerCharge"))
            if not np.isnan(q):  # skip NaN
                charges.append((atom.GetIdx(), atom.GetSymbol(), q))
        except Exception:
            pass

    if charges:
        max_charge = max(charges, key=lambda x: x[2])
        min_charge = min(charges, key=lambda x: x[2])
        charge_range = max_charge[2] - min_charge[2]
    else:
        max_charge = (0, "?", float("nan"))
        min_charge = (0, "?", float("nan"))
        charge_range = float("nan")

    # EState
    if cached and "MaxEStateIndex" in cached:
        max_estate = cached["MaxEStateIndex"]
    else:
        max_estate = float(Descriptors.MaxEStateIndex(mol))
    if cached and "MinEStateIndex" in cached:
        min_estate = cached["MinEStateIndex"]
    else:
        min_estate = float(Descriptors.MinEStateIndex(mol))

    lines = [
        "Charge & EState Properties:",
        f"- Max Gasteiger charge: {max_charge[2]:.4f} (atom {max_charge[0]}, {max_charge[1]})",
        f"- Min Gasteiger charge: {min_charge[2]:.4f} (atom {min_charge[0]}, {min_charge[1]})",
        f"- Charge polarization: {charge_range:.4f}",
        f"- Max EState index (nucleophilic proxy): {max_estate:.4f}",
        f"- Min EState index (electrophilic proxy): {min_estate:.4f}",
    ]
    return "\n".join(lines)


def _generate_coords(smiles: str):
    """Generate 3D coordinates from SMILES. Returns (atomic_numbers, positions)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("Conformer embedding failed")

    # Minimize
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
    except Exception:
        pass

    conf = mol.GetConformer(cid)
    n_atoms = mol.GetNumAtoms()
    numbers = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)])
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n_atoms)])
    return numbers, positions


def _compute_xtb_properties(smiles: str) -> str:
    """
    Run GFN2-xTB via the Python API and extract electronic properties.
    """
    from xtb.interface import Calculator, Param

    numbers, positions = _generate_coords(smiles)

    # positions must be in Bohr for xtb
    ANGSTROM_TO_BOHR = 1.8897259886
    positions_bohr = positions * ANGSTROM_TO_BOHR

    calc = Calculator(Param.GFN2xTB, numbers, positions_bohr)
    calc.set_verbosity(0)
    res = calc.singlepoint()

    # Orbital eigenvalues (in Hartree -> eV)
    HARTREE_TO_EV = 27.211386245988
    orbital_eigenvalues = res.get_orbital_eigenvalues()
    occupations = res.get_orbital_occupations()

    homo_idx = None
    lumo_idx = None
    if orbital_eigenvalues is not None and occupations is not None:
        for i in range(len(occupations)):
            if occupations[i] > 0.5:
                homo_idx = i
        if homo_idx is not None and homo_idx + 1 < len(orbital_eigenvalues):
            lumo_idx = homo_idx + 1

    if homo_idx is not None:
        homo = orbital_eigenvalues[homo_idx] * HARTREE_TO_EV
        lumo = orbital_eigenvalues[lumo_idx] * HARTREE_TO_EV if lumo_idx is not None else float("nan")
        gap = lumo - homo if lumo_idx is not None else float("nan")
    else:
        homo = lumo = gap = float("nan")

    # Dipole moment (Debye — xtb returns in e·Bohr, 1 e·Bohr = 2.5417 Debye)
    dipole_vec = res.get_dipole()
    EBOHR_TO_DEBYE = 2.5417464519
    dipole = float(np.linalg.norm(dipole_vec)) * EBOHR_TO_DEBYE if dipole_vec is not None else float("nan")

    # Total energy (Hartree)
    energy = res.get_energy()

    # Electrophilicity index: ω = μ² / (2η)
    mu = (homo + lumo) / 2.0
    eta = (lumo - homo) / 2.0
    electrophilicity = mu ** 2 / (2 * eta) if eta > 0 else float("nan")

    lines = [
        "Quantum Electronic Properties (GFN2-xTB):",
        f"- HOMO energy: {homo:.4f} eV",
        f"- LUMO energy: {lumo:.4f} eV",
        f"- HOMO-LUMO gap: {gap:.4f} eV",
        f"- Dipole moment: {dipole:.4f} Debye",
        f"- Electrophilicity index (ω): {electrophilicity:.4f} eV",
        f"- Total energy: {energy:.6f} Hartree",
    ]
    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_electronic_properties",
        "description": (
            "Compute electronic properties: HOMO/LUMO energies and gap, dipole moment, "
            "electrophilicity index. Uses GFN2-xTB semi-empirical quantum chemistry if "
            "available, otherwise falls back to RDKit Gasteiger charges. HOMO-LUMO gap "
            "indicates reactivity (small gap → more reactive → potential mutagenicity). "
            "Dipole moment correlates with solubility and membrane permeability."
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
