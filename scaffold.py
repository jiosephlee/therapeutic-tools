"""
Tool 12: Scaffold Decomposition — Murcko scaffold and generic framework extraction.

Strips side chains to reveal the core ring+linker skeleton of a molecule.
Two levels of abstraction:
  - Murcko scaffold: retains atom types and bond orders
  - Generic scaffold: all atoms → carbon, all bonds → single (pure topology)
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


def get_scaffold(smiles: str) -> str:
    """
    Extract the Murcko scaffold and generic framework of a molecule.

    The Murcko scaffold keeps all ring systems and the linker chains
    connecting them, but strips terminal side chains.  The generic
    framework further abstracts atom types to carbon and bond orders
    to single bonds, giving a pure topological skeleton.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with scaffold analysis.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = _mol_from_smiles(smiles)

    # Murcko scaffold (retains heteroatoms and bond orders)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    murcko_smi = Chem.MolToSmiles(core) if core.GetNumAtoms() > 0 else ""

    # Generic framework (all atoms → C, all bonds → single)
    if core.GetNumAtoms() > 0:
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        generic_smi = Chem.MolToSmiles(generic)
    else:
        generic_smi = ""

    # Compute how much of the molecule IS the scaffold
    mol_heavy = mol.GetNumHeavyAtoms()
    scaffold_heavy = core.GetNumHeavyAtoms()
    scaffold_fraction = scaffold_heavy / mol_heavy if mol_heavy > 0 else 0.0

    # Count ring systems in the scaffold
    ring_info = core.GetRingInfo()
    n_rings = ring_info.NumRings() if core.GetNumAtoms() > 0 else 0

    # Side chain atom count
    side_chain_atoms = mol_heavy - scaffold_heavy

    lines = [
        "Scaffold Analysis:",
        f"- Murcko scaffold: {murcko_smi}" if murcko_smi else "- Murcko scaffold: (acyclic molecule — no scaffold)",
        f"- Generic framework: {generic_smi}" if generic_smi else "- Generic framework: (none)",
        f"- Scaffold atoms: {scaffold_heavy} / {mol_heavy} ({scaffold_fraction:.0%} of molecule)",
        f"- Side-chain atoms: {side_chain_atoms}",
        f"- Rings in scaffold: {n_rings}",
    ]
    return "\n".join(lines)


def murcko_scaffold_smiles(smiles: str) -> str:
    """Return just the Murcko scaffold SMILES (for programmatic use by other tools)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = _mol_from_smiles(smiles)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core.GetNumAtoms() > 0 else ""


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_scaffold",
        "description": (
            "Extract the Murcko scaffold (core ring+linker skeleton with side chains "
            "stripped) and generic framework (pure topological skeleton). Shows what "
            "fraction of the molecule is scaffold vs. side chains. Use this to identify "
            "the core structural class of a molecule and compare scaffolds across molecules."
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
