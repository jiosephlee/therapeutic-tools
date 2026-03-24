"""
Tool 3: Ring System Analysis — ring topology, aromaticity, special features.

Compact lossless output: per-system descriptions, one-line ring counts,
conditional flags for PAH/macrocycle/spiro/bridgehead, "Not detected" for absent features.
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
    Analyze ring topology, aromaticity, and structural features.

    Returns a compact lossless report with per-system descriptions,
    ring type counts, and conditional flags for special features
    (PAH, macrocycle, spiro, bridgehead).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Multi-line formatted string with ring system analysis.
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski, rdMolDescriptors
    from . import metadata_cache

    mol = _mol_from_smiles(smiles)
    cached = metadata_cache.lookup_row(smiles)

    def _c(prop, compute_fn):
        if cached and prop in cached:
            return cached[prop]
        return compute_fn()

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    bond_rings = ring_info.BondRings()

    if not atom_rings:
        return "Ring Systems:\n- No rings detected."

    # Build fused ring systems via BFS on shared-bond adjacency
    bond_sets = [set(br) for br in bond_rings]
    n = len(atom_rings)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if bond_sets[i] & bond_sets[j]:
                adj[i].add(j)
                adj[j].add(i)

    visited = set()
    components = []
    for seed in range(n):
        if seed in visited:
            continue
        component = []
        queue = [seed]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node] - visited)
        components.append(component)

    # Analyze each system
    system_descs = []
    largest_aromatic_system = 0

    for component in components:
        system_atoms = set()
        ring_sizes = []
        aromatic_count = 0

        for ring_idx in component:
            ring_atom_indices = atom_rings[ring_idx]
            system_atoms.update(ring_atom_indices)
            ring_sizes.append(len(ring_atom_indices))
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atom_indices):
                aromatic_count += 1

        heteroatom_counts = {}
        for atom_idx in system_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() != 6:
                sym = atom.GetSymbol()
                heteroatom_counts[sym] = heteroatom_counts.get(sym, 0) + 1

        num_rings = len(component)
        largest_aromatic_system = max(largest_aromatic_system, aromatic_count)

        # Build size string
        size_str = "+".join(str(s) for s in sorted(ring_sizes))

        # Build description
        parts = [f"{size_str} membered"]
        if aromatic_count == num_rings:
            parts.append("aromatic")
        elif aromatic_count > 0:
            parts.append(f"{aromatic_count} aromatic")
        else:
            parts.append("non-aromatic")

        if heteroatom_counts:
            het_parts = [f"{v} {k}" for k, v in sorted(heteroatom_counts.items())]
            parts.append(", ".join(het_parts))
        else:
            parts.append("no heteroatoms")

        desc = ", ".join(parts)
        if num_rings > 1:
            desc = f"fused ({desc})"
        system_descs.append((num_rings, aromatic_count, desc))

    # Sort largest first
    system_descs.sort(key=lambda s: (-s[0], -s[1]))

    # Ring type counts
    total_rings = int(_c("RingCount", lambda: float(Lipinski.RingCount(mol))))
    aromatic_rings = int(_c("NumAromaticRings", lambda: float(Lipinski.NumAromaticRings(mol))))
    aliphatic_rings = int(_c("NumAliphaticRings", lambda: float(Lipinski.NumAliphaticRings(mol))))
    saturated_rings = int(_c("NumSaturatedRings", lambda: float(Lipinski.NumSaturatedRings(mol))))
    heterocycles = int(_c("NumHeterocycles", lambda: float(Lipinski.NumHeterocycles(mol))))

    # Aromaticity
    n_aromatic_atoms = int(_c("NumAromaticAtoms", lambda: float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))))
    n_atoms = mol.GetNumAtoms()

    # Special features
    has_pah = largest_aromatic_system >= 3
    all_ring_sizes = [len(ring) for ring in atom_rings]
    num_macrocycles = sum(1 for size in all_ring_sizes if size >= 12)
    spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    bridgehead = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

    # Build output
    lines = ["Ring Systems:"]

    # Per-system descriptions
    if len(system_descs) == 1:
        lines.append(f"- 1 system: {system_descs[0][2]}")
    else:
        lines.append(f"- {len(system_descs)} systems:")
        for _, _, desc in system_descs:
            lines.append(f"  - {desc}")

    # Ring type counts — one line
    lines.append(f"- Ring counts: {total_rings} total, {aromatic_rings} aromatic, {aliphatic_rings} aliphatic, {saturated_rings} saturated")

    # Heterocycles
    lines.append(f"- Heterocycles: {heterocycles}")

    # Aromaticity
    lines.append(f"- Aromatic atoms: {n_aromatic_atoms} / {n_atoms}")

    # Conditional special feature lines (only when present)
    if has_pah:
        lines.append(f"- PAH-like system: {largest_aromatic_system} fused aromatic rings")
    if num_macrocycles > 0:
        lines.append(f"- Macrocycles: {num_macrocycles} (largest ring: {max(all_ring_sizes)}-membered)")
    if spiro > 0:
        lines.append(f"- Spiro centers: {spiro}")
    if bridgehead > 0:
        lines.append(f"- Bridgehead atoms: {bridgehead}")

    # Not detected line for absent features
    absent = []
    if not has_pah:
        absent.append("PAH")
    if num_macrocycles == 0:
        absent.append("macrocycle")
    if spiro == 0:
        absent.append("spiro")
    if bridgehead == 0:
        absent.append("bridgehead")
    if absent:
        lines.append(f"- Not detected: {', '.join(absent)}")

    return "\n".join(lines)


TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "analyze_ring_systems",
        "description": (
            "Analyze ring topology and aromaticity. "
            "Returns per-system descriptions with ring sizes and heteroatom content, "
            "ring type counts (aromatic/aliphatic/saturated/heterocyclic), "
            "aromaticity metrics, and flags for PAH, macrocycle, spiro, and bridgehead features."
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
