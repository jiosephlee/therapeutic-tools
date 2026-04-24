"""
Rerank coarse ChEMBL candidates with the project's custom Morgan + FeatureMorgan
weighted Tanimoto and write the final top-10 neighbors for official_v15.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from chembl_cache_utils import CHEMBL_OFFICIAL_V15_DIR


def make_generator(use_features: bool, radius: int, nbits: int):
    if use_features:
        inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    else:
        inv_gen = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nbits,
        atomInvariantsGenerator=inv_gen,
    )


def compute_fp(smiles: str, generator) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprintAsNumPy(mol).astype(np.float32)


def tanimoto_against(
    query_fp: np.ndarray,
    ref_fps: np.ndarray,
    ref_popcnt: np.ndarray,
) -> np.ndarray:
    a_and_b = ref_fps @ query_fp
    a_bits = query_fp.sum()
    denom = a_bits + ref_popcnt.astype(np.float32) - a_and_b
    return np.where(denom > 0, a_and_b / denom, 0.0)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_coarse_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = build_args()
    coarse_path = CHEMBL_OFFICIAL_V15_DIR / f"coarse_topk_k{args.top_k}.jsonl"
    subset_fp_path = CHEMBL_OFFICIAL_V15_DIR / f"subset_k{args.top_k}_fingerprints.npz"
    subset_meta_path = CHEMBL_OFFICIAL_V15_DIR / f"subset_k{args.top_k}_metadata.csv"
    out_path = CHEMBL_OFFICIAL_V15_DIR / f"top{args.final_k}_weighted_neighbors_k{args.top_k}.jsonl"

    if not coarse_path.exists():
        raise FileNotFoundError(f"Missing coarse candidates: {coarse_path}")
    if not subset_fp_path.exists() or not subset_meta_path.exists():
        raise FileNotFoundError(
            f"Missing subset inputs: {subset_fp_path} and/or {subset_meta_path}"
        )
    if args.overwrite and out_path.exists():
        out_path.unlink()

    coarse_records = load_coarse_records(coarse_path)
    if args.limit is not None:
        coarse_records = coarse_records[: args.limit]

    data = np.load(subset_fp_path, allow_pickle=True)
    molregnos = data["molregnos"].astype(np.int64)
    chembl_ids = data["chembl_ids"]
    canonical_smiles = data["canonical_smiles"]
    morgan_fps = data["morgan_fps"].astype(np.float32)
    feat_fps = data["feat_morgan_fps"].astype(np.float32)
    morgan_popcnt = data["morgan_popcnt"].astype(np.float32)
    feat_popcnt = data["feat_popcnt"].astype(np.float32)
    meta_df = pd.read_csv(subset_meta_path)

    idx_by_molregno = {int(m): i for i, m in enumerate(molregnos)}
    meta_by_molregno = {
        int(row["molregno"]): row.to_dict() for _, row in meta_df.iterrows()
    }

    morgan_gen = make_generator(False, args.radius, args.fp_size)
    feat_gen = make_generator(True, args.radius, args.fp_size)

    completed = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                query_smiles = payload.get("query_smiles")
                if isinstance(query_smiles, str):
                    completed.add(query_smiles)

    with out_path.open("a") as out_f:
        for i, record in enumerate(coarse_records, 1):
            query_smiles = record["query_smiles"]
            if query_smiles in completed:
                continue

            query_morgan = compute_fp(query_smiles, morgan_gen)
            query_feat = compute_fp(query_smiles, feat_gen)
            if query_morgan is None or query_feat is None:
                continue

            candidate_molregnos = [
                int(n["molregno"])
                for n in record["neighbors"]
                if int(n["molregno"]) in idx_by_molregno
            ]
            if not candidate_molregnos:
                continue
            candidate_idx = np.array([idx_by_molregno[m] for m in candidate_molregnos], dtype=np.int64)
            cand_morgan = morgan_fps[candidate_idx]
            cand_feat = feat_fps[candidate_idx]
            morgan_scores = tanimoto_against(query_morgan, cand_morgan, morgan_popcnt[candidate_idx])
            feat_scores = tanimoto_against(query_feat, cand_feat, feat_popcnt[candidate_idx])
            weighted = 0.8 * morgan_scores + 0.2 * feat_scores
            order = np.argsort(weighted)[::-1][: args.final_k]

            coarse_by_molregno = {
                int(n["molregno"]): float(n["coarse_morgan_tanimoto"]) for n in record["neighbors"]
            }
            neighbors = []
            for rank, local_idx in enumerate(order, 1):
                molregno = candidate_molregnos[int(local_idx)]
                meta = dict(meta_by_molregno.get(molregno, {}))
                neighbors.append(
                    {
                        "rank": rank,
                        "molregno": molregno,
                        "chembl_id": str(meta.get("chembl_id", chembl_ids[candidate_idx[local_idx]])),
                        "canonical_smiles": str(
                            meta.get("canonical_smiles", canonical_smiles[candidate_idx[local_idx]])
                        ),
                        "coarse_morgan_tanimoto": coarse_by_molregno.get(molregno),
                        "morgan_tanimoto": float(morgan_scores[local_idx]),
                        "feature_morgan_tanimoto": float(feat_scores[local_idx]),
                        "weighted_tanimoto": float(weighted[local_idx]),
                        "pref_name": meta.get("pref_name"),
                        "max_phase": meta.get("max_phase"),
                        "first_approval": meta.get("first_approval"),
                        "black_box_warning": meta.get("black_box_warning"),
                        "dosed_ingredient": meta.get("dosed_ingredient"),
                        "molecule_type": meta.get("molecule_type"),
                        "availability_type": meta.get("availability_type"),
                        "withdrawn_flag": meta.get("withdrawn_flag"),
                        "mechanism_summary": meta.get("mechanism_summary"),
                        "indication_summary": meta.get("indication_summary"),
                        "warning_summary": meta.get("warning_summary"),
                    }
                )

            payload = {
                "query_smiles": query_smiles,
                "canonical_smiles": record.get("canonical_smiles"),
                "tasks": record.get("tasks", []),
                "neighbors": neighbors,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"reranked {i}/{len(coarse_records)} queries")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
