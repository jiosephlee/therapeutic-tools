"""
ATTNSOM single-molecule inference wrapper.

Usage:
    from ATTNSOM.inference import ATTNSOMPredictor
    predictor = ATTNSOMPredictor("/path/to/attnsom_checkpoint.pt")
    results = predictor.predict("CCO", cyp="2D6")
"""

import os
import torch
from functools import lru_cache
from typing import List, Tuple, Optional

from rdkit import Chem

# Use absolute imports so this works both standalone and as a subpackage
try:
    from .model import GraphCliffMultiRegressor
    from .dataset_utils import mol_to_graph
except ImportError:
    from model import GraphCliffMultiRegressor
    from dataset_utils import mol_to_graph


CYP_LIST = ['1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4']

# 10-fold CV performance metrics per isoform (from training with TDC exclusion)
CYP_METRICS = {
    '1A2':  {'f1': 0.694, 'top3': 0.896},
    '2A6':  {'f1': 0.796, 'top3': 0.963},
    '2B6':  {'f1': 0.842, 'top3': 0.948},
    '2C8':  {'f1': 0.800, 'top3': 0.889},
    '2C9':  {'f1': 0.702, 'top3': 0.983},
    '2C19': {'f1': 0.779, 'top3': 0.949},
    '2D6':  {'f1': 0.707, 'top3': 0.920},
    '2E1':  {'f1': 0.750, 'top3': 0.953},
    '3A4':  {'f1': 0.611, 'top3': 0.831},
}


class ATTNSOMPredictor:
    """Load a trained ATTNSOM checkpoint and predict SoM for single molecules."""

    def __init__(self, checkpoint_path: str, device: str = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.cyp_list = ckpt.get("cyp_list", CYP_LIST)
        self.cyp2idx = {c: i for i, c in enumerate(self.cyp_list)}
        atom_in_dim = ckpt["atom_in_dim"]
        edge_dim = ckpt["edge_dim"]
        args = ckpt.get("args", {})

        self.model = GraphCliffMultiRegressor(
            atom_in_dim=atom_in_dim,
            edge_dim=edge_dim,
            hidden_size=args.get("hidden_size", 256),
            num_layers=args.get("num_layers", 4),
            groups=4,
            mid_K=3,
            dropout=0.0,
            cyp_names=self.cyp_list,
            num_attn_heads=args.get("num_attn_heads", 4),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._cache = {}  # (canonical_smiles, cyp_key) -> result dict

    def predict(
        self, smiles: str, cyp: Optional[str] = None, threshold: float = 0.5
    ) -> dict:
        """
        Predict SoM for a single molecule. Results are cached by (canonical SMILES, cyp).

        Args:
            smiles: SMILES string.
            cyp: CYP isoform (e.g. '2D6'). If None, runs all 9 isoforms.
            threshold: Probability threshold for positive SoM call.

        Returns:
            dict with keys: smiles, predictions (list of per-CYP results).
            Each CYP result has: cyp, atoms (list of {idx, symbol, prob, is_som}).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        canon_smi = Chem.MolToSmiles(mol)
        cache_key = (canon_smi, cyp or "all")
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Re-parse canonical SMILES so atom ordering is consistent
        mol = Chem.MolFromSmiles(canon_smi)

        data, _ = mol_to_graph(mol, canon_smi)

        cyps_to_run = [cyp] if cyp else self.cyp_list
        results = {"smiles": canon_smi, "predictions": []}

        for c in cyps_to_run:
            if c not in self.cyp2idx:
                continue
            cidx = self.cyp2idx[c]

            # Set CYP index for all atoms
            data.cyp_idx = torch.full((data.num_nodes,), cidx, dtype=torch.long)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

            # Create dummy som_annotations/som_mask (not used in inference)
            num_cyps = len(self.cyp_list)
            data.som_annotations = torch.zeros(data.num_nodes, num_cyps)
            data.som_mask = torch.zeros(data.num_nodes, num_cyps)

            batch = data.to(self.device)
            with torch.no_grad():
                logits, _, attn_weights = self.model(batch)
                probs = torch.sigmoid(logits).cpu().tolist()

            atoms = []
            for i in range(data.num_nodes):
                atom = mol.GetAtomWithIdx(i)
                atoms.append({
                    "idx": i,
                    "symbol": atom.GetSymbol(),
                    "prob": round(probs[i], 4),
                    "is_som": probs[i] > threshold,
                })

            results["predictions"].append({
                "cyp": c,
                "atoms": atoms,
                "top_sites": sorted(atoms, key=lambda a: a["prob"], reverse=True)[:5],
            })

        self._cache[cache_key] = results
        return results


def format_prediction(result: dict) -> str:
    """Format prediction result as a human-readable string."""
    lines = [f"ATTNSOM CYP450 Site-of-Metabolism Prediction", ""]

    for pred in result["predictions"]:
        cyp = pred["cyp"]
        top = pred["top_sites"]
        som_atoms = [a for a in pred["atoms"] if a["is_som"]]

        metrics = CYP_METRICS.get(cyp, {})
        f1 = metrics.get('f1', 'N/A')
        top3 = metrics.get('top3', 'N/A')
        f1_str = f"{f1:.3f}" if isinstance(f1, float) else f1
        top3_str = f"{top3:.3f}" if isinstance(top3, float) else top3
        lines.append(f"  CYP{cyp} (model F1={f1_str}, top-3 acc={top3_str}):")
        for rank, a in enumerate(top, 1):
            marker = " *SoM*" if a["is_som"] else ""
            lines.append(f"    {rank}. Atom {a['idx']} ({a['symbol']}), p = {a['prob']:.3f}{marker}")
        if som_atoms:
            lines.append(f"    → {len(som_atoms)} predicted SoM site(s)")
        else:
            lines.append(f"    → No high-confidence SoM sites")
        lines.append("")

    n_total_som = sum(
        len([a for a in p["atoms"] if a["is_som"]]) for p in result["predictions"]
    )
    lines.append(f"Summary: {n_total_som} total SoM sites across {len(result['predictions'])} CYP isoform(s)")
    return "\n".join(lines)
