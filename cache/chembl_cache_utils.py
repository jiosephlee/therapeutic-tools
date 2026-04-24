from __future__ import annotations

import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from rdkit import Chem


CACHE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CACHE_DIR.parent.parent.parent.parent

CHEMBL_CACHE_DIR = CACHE_DIR / "chembl"
CHEMBL_RAW_DIR = CHEMBL_CACHE_DIR / "raw"
CHEMBL_OFFICIAL_V15_DIR = CHEMBL_CACHE_DIR / "official_v15"
CHEMBL_SQLITE_DIR = CHEMBL_CACHE_DIR / "sqlite"

CHEMBL_H5_PATH = CHEMBL_RAW_DIR / "chembl_36.h5"
CHEMBL_CHEMREPS_PATH = CHEMBL_RAW_DIR / "chembl_36_chemreps.txt.gz"
CHEMBL_SQLITE_TAR_PATH = CHEMBL_RAW_DIR / "chembl_36_sqlite.tar.gz"

OFFICIAL_V15_DATASET_DIR = REPO_ROOT / "data" / "tdc" / "official_v15_dataset"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_official_v15_queries() -> List[Dict[str, object]]:
    rows_by_smiles: Dict[str, Dict[str, object]] = {}
    tasks_by_smiles: dict[str, set[str]] = defaultdict(set)

    for csv_path in sorted(OFFICIAL_V15_DATASET_DIR.glob("*/*.csv")):
        task = csv_path.parent.name
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles = row.get("Drug") or row.get("smiles") or row.get("SMILES")
                if not smiles:
                    continue
                if smiles not in rows_by_smiles:
                    rows_by_smiles[smiles] = {
                        "query_smiles": smiles,
                        "canonical_smiles": canonicalize_smiles(smiles),
                    }
                tasks_by_smiles[smiles].add(task)

    output: List[Dict[str, object]] = []
    for smiles in sorted(rows_by_smiles):
        row = dict(rows_by_smiles[smiles])
        row["tasks"] = sorted(tasks_by_smiles[smiles])
        output.append(row)
    return output


def load_completed_queries(jsonl_path: Path) -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            query_smiles = payload.get("query_smiles")
            if isinstance(query_smiles, str):
                completed.add(query_smiles)
    return completed


def write_jsonl(path: Path, records: Iterable[dict], mode: str = "a") -> None:
    ensure_directory(path.parent)
    with path.open(mode) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_sqlite_db_path() -> Path | None:
    if not CHEMBL_SQLITE_DIR.exists():
        return None
    for suffix in ("*.db", "*.sqlite", "*.sqlite3"):
        matches = sorted(CHEMBL_SQLITE_DIR.rglob(suffix))
        if matches:
            return matches[0]
    return None


def extract_sqlite_db_if_needed() -> Path:
    existing = resolve_sqlite_db_path()
    if existing is not None:
        return existing
    if not CHEMBL_SQLITE_TAR_PATH.exists():
        raise FileNotFoundError(
            f"Missing SQLite archive at {CHEMBL_SQLITE_TAR_PATH}. Download it first."
        )

    ensure_directory(CHEMBL_SQLITE_DIR)
    with tarfile.open(CHEMBL_SQLITE_TAR_PATH, "r:gz") as tf:
        db_members = [
            member
            for member in tf.getmembers()
            if member.isfile()
            and (
                member.name.endswith(".db")
                or member.name.endswith(".sqlite")
                or member.name.endswith(".sqlite3")
            )
        ]
        if not db_members:
            raise FileNotFoundError(
                f"No SQLite database file found inside {CHEMBL_SQLITE_TAR_PATH}"
            )
        tf.extractall(CHEMBL_SQLITE_DIR, members=db_members)

    extracted = resolve_sqlite_db_path()
    if extracted is None:
        raise FileNotFoundError(
            f"Failed to extract SQLite database from {CHEMBL_SQLITE_TAR_PATH}"
        )
    return extracted


def chunked(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
