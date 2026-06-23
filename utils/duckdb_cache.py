"""DuckDB-backed therapeutic feature cache lookups."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .. import paths

PACKAGE_DIR = paths.PACKAGE_DIR
DEFAULT_DUCKDB_PATH = paths.cache_path("therapeutic_tools.duckdb")
MINIMOL_METHOD = "MiniMol + Ridge (AqSolDB)"


def duckdb_only_mode() -> bool:
    return os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND", "").lower() == "duckdb_only"


def duckdb_disabled() -> bool:
    return os.environ.get("THERAPEUTIC_TOOLS_DISABLE_DUCKDB_CACHE", "").lower() in {"1", "true", "yes", "on"}


def cache_path() -> Path | None:
    path = paths.duckdb_path()
    return path if path.exists() else None


def feature_set_key(feature_names: Iterable[str]) -> str:
    return ",".join(str(name) for name in feature_names)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _canonicalize(smiles: str) -> str | None:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def _has_minimol_provenance(provenance: Any) -> bool:
    data = _json_object(provenance)
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, dict):
        return False
    minimol = endpoints.get("minimol_solubility_log_s")
    return isinstance(minimol, dict) and minimol.get("method") == MINIMOL_METHOD


def _connect(path: Path):
    try:
        import duckdb
    except Exception as exc:
        if duckdb_only_mode():
            raise RuntimeError("DuckDB cache is required but duckdb is not importable") from exc
        return None
    return duckdb.connect(str(path), read_only=True)


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def lookup_full_call_string(
    smiles: str,
    feature_names: Iterable[str],
    *,
    allow_canonical_fallback: bool = True,
) -> str | None:
    """Return feature text through the DuckDB-first facade.

    The facade performs raw-then-canonical lookup, computes missing primitives
    through configured runtime subprocesses, writes primitive rows, renders
    bundle text, and stores the derived full-call row.
    """
    if duckdb_disabled():
        return None
    try:
        from ..api import get_feature_text

        return get_feature_text(smiles, feature_names)
    except Exception:
        if duckdb_only_mode():
            raise
        return None


def lookup_full_call_string_legacy(
    smiles: str,
    feature_names: Iterable[str],
    *,
    allow_canonical_fallback: bool = True,
) -> str | None:
    """Legacy read-only lookup used only by explicit compatibility callers."""
    if duckdb_disabled():
        return None
    path = cache_path()
    if path is None or not path.exists():
        if duckdb_only_mode():
            raise RuntimeError("DuckDB-only mode is enabled but no DuckDB cache path is configured")
        return None

    names = tuple(str(name) for name in feature_names)
    keys = [str(smiles)]
    if allow_canonical_fallback:
        canonical = _canonicalize(str(smiles))
        if canonical and canonical not in keys:
            keys.append(canonical)

    conn = _connect(path)
    if conn is None:
        return None
    try:
        if _table_exists(conn, "full_call_strings"):
            key = feature_set_key(names)
            for candidate in keys:
                row = conn.execute(
                    """
                    SELECT feature_text, provenance_json
                    FROM full_call_strings
                    WHERE status = 'ok'
                      AND feature_set_key = ?
                      AND raw_smiles = ?
                    """,
                    [key, candidate],
                ).fetchone()
                if row is None and allow_canonical_fallback:
                    row = conn.execute(
                        """
                        SELECT feature_text, provenance_json
                        FROM full_call_strings
                        WHERE status = 'ok'
                          AND feature_set_key = ?
                          AND canonical_smiles = ?
                        """,
                        [key, candidate],
                    ).fetchone()
                if row is not None:
                    text, provenance = row
                    if "ionization_and_solubility" in names and not _has_minimol_provenance(provenance):
                        raise RuntimeError(f"DuckDB full_call_strings row lacks MiniMol provenance for {smiles!r}")
                    return str(text or "")

        if _table_exists(conn, "semantic_strings"):
            for candidate in keys:
                rows = conn.execute(
                    """
                    SELECT feature_name, text, provenance_json
                    FROM semantic_strings
                    WHERE status = 'ok'
                      AND raw_smiles = ?
                      AND feature_name IN (SELECT * FROM UNNEST(?))
                    """,
                    [candidate, list(names)],
                ).fetchall()
                if len(rows) != len(names) and allow_canonical_fallback:
                    rows = conn.execute(
                        """
                        SELECT feature_name, text, provenance_json
                        FROM semantic_strings
                        WHERE status = 'ok'
                          AND canonical_smiles = ?
                          AND feature_name IN (SELECT * FROM UNNEST(?))
                        """,
                        [candidate, list(names)],
                    ).fetchall()
                if len(rows) != len(names):
                    continue
                by_name = {str(name): (str(text or ""), provenance) for name, text, provenance in rows}
                if "ionization_and_solubility" in names:
                    _, provenance = by_name["ionization_and_solubility"]
                    if not _has_minimol_provenance(provenance):
                        raise RuntimeError(f"DuckDB semantic_strings row lacks MiniMol provenance for {smiles!r}")
                return "\n\n".join(by_name[name][0] for name in names if by_name[name][0])
    finally:
        conn.close()

    if duckdb_only_mode():
        raise RuntimeError(f"DuckDB-only mode is enabled but no cached full-call string was found for {smiles!r}")
    return None
