"""
Tool 3: Ring System Analysis — ring topology, aromaticity, shape indices.

Includes fused ring system analysis, ring type breakdown,
aromaticity metrics, and topological shape indices.
"""

from typing import Dict, Any


def _mol_from_smiles(smiles: str):
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def analyze_ring_systems(smiles: str) -> str:
    """
    Analyze ring topology, aromaticity, and shape indices.

    Includes fused ring system analysis, ring type breakdown,
    aromaticity metrics, and topological shape indices.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with ring system analysis.
    """
    from rdkit.Chem import Lipinski
    from .legacy_tools.RDKit_tools import analyze_ring_systems as _analyze_ring_systems_impl
    from . import metadata_cache

    mol = _mol_from_smiles(smiles)
    cached = metadata_cache.lookup_row(smiles)

    def _c(prop, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    # Core ring system analysis (fused systems, PAH, macrocycles, etc.)
    ring_report = _analyze_ring_systems_impl(smiles)

    # Ring type counts
    total_rings = int(_c("RingCount", lambda: float(Lipinski.RingCount(mol))))
    aromatic_rings = int(_c("NumAromaticRings", lambda: float(Lipinski.NumAromaticRings(mol))))
    aliphatic_rings = int(_c("NumAliphaticRings", lambda: float(Lipinski.NumAliphaticRings(mol))))
    saturated_rings = int(_c("NumSaturatedRings", lambda: float(Lipinski.NumSaturatedRings(mol))))
    heterocycles = int(_c("NumHeterocycles", lambda: float(Lipinski.NumHeterocycles(mol))))
    aromatic_heterocycles = int(_c("NumAromaticHeterocycles", lambda: float(Lipinski.NumAromaticHeterocycles(mol))))
    aliphatic_heterocycles = int(_c("NumAliphaticHeterocycles", lambda: float(Lipinski.NumAliphaticHeterocycles(mol))))
    saturated_heterocycles = int(_c("NumSaturatedHeterocycles", lambda: float(Lipinski.NumSaturatedHeterocycles(mol))))

    # Aromaticity metrics
    n_aromatic_atoms = int(_c("NumAromaticAtoms", lambda: float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))))
    n_atoms = mol.GetNumAtoms()
    frac_aromatic = _c("FractionAromaticAtoms", lambda: (n_aromatic_atoms / n_atoms) if n_atoms > 0 else 0.0)

    lines = [
        ring_report,
        "",
        "Ring Type Counts:",
        f"- Total rings: {total_rings}",
        f"- Aromatic: {aromatic_rings}, Aliphatic: {aliphatic_rings}, Saturated: {saturated_rings}",
        f"- Heterocycles: {heterocycles} (aromatic: {aromatic_heterocycles}, aliphatic: {aliphatic_heterocycles}, saturated: {saturated_heterocycles})",
        "",
        "Aromaticity:",
        f"- Aromatic atoms: {n_aromatic_atoms} / {n_atoms} ({frac_aromatic:.4f})",
    ]
    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_ring_systems",
        "description": (
            "Analyze ring topology and aromaticity. "
            "Includes fused ring system analysis, PAH detection (≥3 fused aromatic rings), "
            "macrocycle detection, ring type breakdown (aromatic/aliphatic/saturated/heterocyclic), "
            "and aromaticity metrics."
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
