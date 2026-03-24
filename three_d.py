"""
Tool 5: 3D Properties — conformational ensemble analysis.

Includes 3D exposed polar surface area (ePSA) and 3D shape (PMI ratios).

Computationally expensive (~seconds) — call only when 3D info is needed.
"""

from typing import Dict, Any


def get_3d_properties(smiles: str) -> str:
    """
    Compute 3D conformational properties from a conformer ensemble.

    Includes:
    - 3D exposed polar surface area (ePSA) with Boltzmann weighting
    - 3D shape: PMI ratios (rod/disc/sphere classification)

    This is computationally expensive (~seconds). Call only when 3D surface
    or shape information is specifically needed beyond 2D TPSA.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with 3D property analysis.
    """
    from .legacy_tools.ePSA_3D import (
        exposed_polar_sasa_ensemble,
        get_3d_exposed_polar_surface,
    )

    from . import metadata_cache

    sections = []

    # ePSA — use cache if available
    cached_psa = metadata_cache.lookup(smiles, "PSA_3D")
    if cached_psa is not None:
        sections.append("3D Polar Surface Area (ePSA):")
        sections.append(f"3D conformation based estimation of PSA: {cached_psa:.2f}")
    else:
        try:
            epsa_result = get_3d_exposed_polar_surface(smiles)
            sections.append("3D Polar Surface Area (ePSA):")
            sections.append(epsa_result if isinstance(epsa_result, str) else str(epsa_result))
        except Exception as e:
            sections.append(f"3D ePSA: Error - {e}")

    # 3D Shape descriptors (PMI expansion)
    try:
        shape_result = _compute_shape_descriptors(smiles)
        sections.append("")
        sections.append(shape_result)
    except Exception as e:
        sections.append(f"\n3D Shape: Error - {e}")

    return "\n".join(sections)


def _compute_shape_descriptors(smiles: str) -> str:
    """
    Compute 3D shape descriptors from the lowest-energy conformer.

    Uses RDKit's Descriptors3D module for PMI (Principal Moments of Inertia)
    shape classification.

    These are cheap to compute once a conformer exists.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    # Generate a single conformer (fast — just need one for shape)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return "3D Shape: Conformer embedding failed"

    # Minimize with MMFF
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
    except Exception:
        pass  # Shape is still useful even without minimization

    # Compute PMI (Principal Moments of Inertia)
    try:
        pmi1 = Descriptors3D.PMI1(mol, confId=cid)
        pmi2 = Descriptors3D.PMI2(mol, confId=cid)
        pmi3 = Descriptors3D.PMI3(mol, confId=cid)

        # Normalized PMI ratios (rod/disc/sphere classification)
        # npr1 = I1/I3, npr2 = I2/I3, where I1 <= I2 <= I3
        npr1 = Descriptors3D.NPR1(mol, confId=cid)  # I1/I3
        npr2 = Descriptors3D.NPR2(mol, confId=cid)  # I2/I3

        # Shape classification based on PMI plot position
        # Rod: npr1 → 0, npr2 → 0  (linear)
        # Disc: npr1 → 0, npr2 → 1  (flat/planar)
        # Sphere: npr1 → 1, npr2 → 1
        if npr1 > 0.6 and npr2 > 0.6:
            shape_class = "sphere-like"
        elif npr1 < 0.3 and npr2 > 0.6:
            shape_class = "disc-like"
        elif npr1 < 0.3 and npr2 < 0.5:
            shape_class = "rod-like"
        else:
            shape_class = "intermediate"
    except Exception:
        pmi1 = pmi2 = pmi3 = npr1 = npr2 = float("nan")
        shape_class = "unknown"

    lines = [
        "3D Shape Descriptors:",
        f"- PMI ratios (npr1, npr2): ({npr1:.4f}, {npr2:.4f}) → {shape_class}",
    ]
    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_3d_properties",
        "description": (
            "Compute 3D conformational properties from a conformer ensemble. "
            "Returns exposed polar surface area (ePSA) and 3D shape (PMI ratios for "
            "rod/disc/sphere classification). "
            "Computationally expensive (~seconds). Use only when 3D surface/shape information "
            "is specifically needed beyond 2D TPSA."
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
