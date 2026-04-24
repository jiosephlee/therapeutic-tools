"""
Build a reduced ChEMBL subset cache after coarse top-K retrieval.

Inputs:
  - therapeutic_tools/cache/chembl/official_v15/coarse_topk_k{K}_union_molregnos.txt
  - ChEMBL SQLite dump

Outputs:
  - therapeutic_tools/cache/chembl/official_v15/subset_k{K}_metadata.csv
  - therapeutic_tools/cache/chembl/official_v15/subset_k{K}_fingerprints.npz
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from chembl_cache_utils import (
    CHEMBL_OFFICIAL_V15_DIR,
    chunked,
    extract_sqlite_db_if_needed,
)


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
    return generator.GetFingerprintAsNumPy(mol).astype(np.uint8)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_union_ids(path: Path) -> list[int]:
    molregnos: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                molregnos.append(int(line))
    return molregnos


def fetch_subset_metadata(conn: sqlite3.Connection, molregnos: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_query = """
        SELECT
            md.molregno,
            md.chembl_id,
            md.pref_name,
            md.max_phase,
            md.first_approval,
            md.black_box_warning,
            md.dosed_ingredient,
            md.molecule_type,
            md.availability_type,
            md.therapeutic_flag,
            md.withdrawn_flag,
            md.natural_product,
            md.chemical_probe,
            mh.parent_molregno,
            mh.active_molregno,
            cs.canonical_smiles,
            cs.standard_inchi_key
        FROM molecule_dictionary md
        LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE md.molregno IN ({placeholders})
    """
    for batch in chunked(molregnos, 900):
        query = base_query.format(placeholders=",".join("?" for _ in batch))
        frames.append(pd.read_sql_query(query, conn, params=[int(x) for x in batch]))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["molregno"]).sort_values("molregno").reset_index(drop=True)


def fetch_grouped_strings(conn: sqlite3.Connection, molregnos: list[int], query_tmpl: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in chunked(molregnos, 900):
        query = query_tmpl.format(placeholders=",".join("?" for _ in batch))
        frames.append(pd.read_sql_query(query, conn, params=[int(x) for x in batch]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = build_args()
    union_path = CHEMBL_OFFICIAL_V15_DIR / f"coarse_topk_k{args.top_k}_union_molregnos.txt"
    if not union_path.exists():
        raise FileNotFoundError(f"Missing coarse union file: {union_path}")

    meta_out = CHEMBL_OFFICIAL_V15_DIR / f"subset_k{args.top_k}_metadata.csv"
    fp_out = CHEMBL_OFFICIAL_V15_DIR / f"subset_k{args.top_k}_fingerprints.npz"
    if not args.overwrite and meta_out.exists() and fp_out.exists():
        print(f"Outputs already exist: {meta_out} and {fp_out}")
        return

    molregnos = sorted(set(load_union_ids(union_path)))
    print(f"Loaded {len(molregnos)} union molregnos from {union_path}")
    db_path = extract_sqlite_db_if_needed()
    print(f"Using SQLite DB: {db_path}")
    conn = sqlite3.connect(str(db_path))

    metadata = fetch_subset_metadata(conn, molregnos)
    mechanism_df = fetch_grouped_strings(
        conn,
        molregnos,
        """
        SELECT
            dm.molregno,
            group_concat(
                DISTINCT COALESCE(dm.mechanism_of_action, '') || '|' ||
                COALESCE(dm.action_type, '') || '|' ||
                COALESCE(td.pref_name, '')
            ) AS mechanism_summary
        FROM drug_mechanism dm
        LEFT JOIN target_dictionary td ON td.tid = dm.tid
        WHERE dm.molregno IN ({placeholders})
        GROUP BY dm.molregno
        """,
    )
    indication_df = fetch_grouped_strings(
        conn,
        molregnos,
        """
        SELECT
            di.molregno,
            group_concat(DISTINCT COALESCE(di.mesh_heading, '')) AS indication_summary
        FROM drug_indication di
        WHERE di.molregno IN ({placeholders})
        GROUP BY di.molregno
        """,
    )
    warning_df = fetch_grouped_strings(
        conn,
        molregnos,
        """
        SELECT
            dw.molregno,
            group_concat(
                DISTINCT COALESCE(dw.warning_type, '') || '|' || COALESCE(dw.warning_class, '')
            ) AS warning_summary
        FROM drug_warning dw
        WHERE dw.molregno IN ({placeholders})
        GROUP BY dw.molregno
        """,
    )
    conn.close()

    for extra in (mechanism_df, indication_df, warning_df):
        if not extra.empty:
            metadata = metadata.merge(extra, on="molregno", how="left")

    morgan_gen = make_generator(False, args.radius, args.fp_size)
    feat_gen = make_generator(True, args.radius, args.fp_size)

    kept_rows = []
    morgan_fps = []
    feat_fps = []
    for _, row in metadata.iterrows():
        smiles = row.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        morgan = compute_fp(smiles, morgan_gen)
        feat = compute_fp(smiles, feat_gen)
        if morgan is None or feat is None:
            continue
        kept_rows.append(row)
        morgan_fps.append(morgan)
        feat_fps.append(feat)

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    print(f"Kept {len(kept_df)} molecules with valid canonical SMILES/fingerprints")
    kept_df.to_csv(meta_out, index=False)

    morgan_arr = np.stack(morgan_fps)
    feat_arr = np.stack(feat_fps)
    np.savez_compressed(
        fp_out,
        molregnos=kept_df["molregno"].to_numpy(dtype=np.int64),
        chembl_ids=kept_df["chembl_id"].fillna("").to_numpy(dtype=object),
        canonical_smiles=kept_df["canonical_smiles"].fillna("").to_numpy(dtype=object),
        morgan_fps=morgan_arr,
        feat_morgan_fps=feat_arr,
        morgan_popcnt=morgan_arr.sum(axis=1).astype(np.int16),
        feat_popcnt=feat_arr.sum(axis=1).astype(np.int16),
    )
    print(f"Wrote {meta_out}")
    print(f"Wrote {fp_out}")


if __name__ == "__main__":
    main()
