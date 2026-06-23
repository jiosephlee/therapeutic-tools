"""DuckDB storage primitives for therapeutic feature values."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .primitive_registry import DEFAULT_PARAMS_JSON, feature_set_key


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DUCKDB_PATH = paths.cache_path("therapeutic_tools.duckdb")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    return value


def params_json(params: dict[str, Any] | None = None) -> str:
    return json.dumps(normalize_for_json(params or {}), sort_keys=True, separators=(",", ":"))


def value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return "json"


def encode_value(value: Any) -> str:
    return json.dumps(normalize_for_json(value), sort_keys=True, separators=(",", ":"))


def decode_value(value_json: Any) -> Any:
    if isinstance(value_json, str):
        return json.loads(value_json)
    return value_json


def duckdb_disabled() -> bool:
    return os.environ.get("THERAPEUTIC_TOOLS_DISABLE_DUCKDB_CACHE", "").lower() in {"1", "true", "yes", "on"}


def duckdb_only_mode() -> bool:
    return os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND", "").lower() == "duckdb_only"


def cache_path() -> Path:
    return paths.duckdb_path()


def canonicalize_smiles(smiles: str) -> dict[str, Any]:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "raw_smiles": smiles,
                "canonical_smiles": None,
                "raw_is_canonical": False,
                "canonicalization_status": "error",
                "canonicalization_error": "invalid SMILES",
            }
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return {
            "raw_smiles": smiles,
            "canonical_smiles": canonical,
            "raw_is_canonical": smiles == canonical,
            "canonicalization_status": "ok",
            "canonicalization_error": None,
        }
    except Exception as exc:
        return {
            "raw_smiles": smiles,
            "canonical_smiles": None,
            "raw_is_canonical": False,
            "canonicalization_status": "error",
            "canonicalization_error": f"{type(exc).__name__}: {exc}",
        }


def connect(path: Path | None = None, *, read_only: bool = False):
    import duckdb

    db_path = path or cache_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only)
    if not read_only:
        ensure_schema(conn)
    return conn


def table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def columns(conn: Any, table_name: str) -> set[str]:
    if not table_exists(conn, table_name):
        return set()
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {str(row[0]) for row in rows}


def add_column(conn: Any, table_name: str, column_name: str, sql_type: str) -> None:
    if column_name not in columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}")


def ensure_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS molecules (
            raw_smiles VARCHAR PRIMARY KEY,
            canonical_smiles VARCHAR,
            raw_is_canonical BOOLEAN,
            canonicalization_status VARCHAR NOT NULL,
            canonicalization_error VARCHAR,
            updated_at_utc VARCHAR NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS primitive_values (
            raw_smiles VARCHAR,
            canonical_smiles VARCHAR,
            raw_is_canonical BOOLEAN,
            canonicalization_status VARCHAR,
            canonicalization_error VARCHAR,
            primitive_name VARCHAR,
            endpoint_version VARCHAR,
            python_executable VARCHAR,
            params_json VARCHAR,
            value_type VARCHAR,
            value_json JSON,
            status VARCHAR,
            error VARCHAR,
            runtime_name VARCHAR,
            method VARCHAR,
            provenance_json JSON,
            updated_at_utc VARCHAR
        )
        """
    )
    for name, typ in (
        ("endpoint_version", "VARCHAR"),
        ("python_executable", "VARCHAR"),
        ("params_json", "VARCHAR"),
        ("value_type", "VARCHAR"),
        ("method", "VARCHAR"),
        ("provenance_json", "JSON"),
    ):
        add_column(conn, "primitive_values", name, typ)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bundle_strings (
            raw_smiles VARCHAR,
            canonical_smiles VARCHAR,
            bundle_name VARCHAR,
            status VARCHAR NOT NULL,
            text VARCHAR,
            error VARCHAR,
            provenance_json JSON,
            updated_at_utc VARCHAR NOT NULL,
            PRIMARY KEY(raw_smiles, bundle_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_strings (
            raw_smiles VARCHAR,
            canonical_smiles VARCHAR,
            feature_name VARCHAR,
            status VARCHAR NOT NULL,
            text VARCHAR,
            error VARCHAR,
            provenance_json JSON,
            updated_at_utc VARCHAR NOT NULL,
            PRIMARY KEY(raw_smiles, feature_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS full_call_strings (
            raw_smiles VARCHAR,
            canonical_smiles VARCHAR,
            feature_set_key VARCHAR,
            status VARCHAR NOT NULL,
            feature_text VARCHAR,
            error VARCHAR,
            provenance_json JSON,
            updated_at_utc VARCHAR NOT NULL,
            PRIMARY KEY(raw_smiles, feature_set_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS build_runs (
            run_id VARCHAR PRIMARY KEY,
            cache_backend VARCHAR NOT NULL,
            feature_set_key VARCHAR NOT NULL,
            source_smiles_count BIGINT NOT NULL,
            record_path_count BIGINT NOT NULL,
            stats_json JSON,
            created_at_utc VARCHAR NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS legacy_audit (
            source_path VARCHAR,
            source_kind VARCHAR,
            decision VARCHAR,
            reason VARCHAR,
            count BIGINT,
            updated_at_utc VARCHAR
        )
        """
    )


def upsert_molecule(conn: Any, smiles_info: dict[str, Any], now: str | None = None) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO molecules(
            raw_smiles, canonical_smiles, raw_is_canonical,
            canonicalization_status, canonicalization_error, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            smiles_info["raw_smiles"],
            smiles_info["canonical_smiles"],
            smiles_info["raw_is_canonical"],
            smiles_info["canonicalization_status"],
            smiles_info["canonicalization_error"],
            now or utc_now(),
        ],
    )


def write_primitive_rows(conn: Any, rows: Iterable[dict[str, Any]]) -> None:
    now = utc_now()
    for row in rows:
        smiles_info = {
            "raw_smiles": row["raw_smiles"],
            "canonical_smiles": row.get("canonical_smiles"),
            "raw_is_canonical": row.get("raw_is_canonical"),
            "canonicalization_status": row.get("canonicalization_status") or "ok",
            "canonicalization_error": row.get("canonicalization_error"),
        }
        upsert_molecule(conn, smiles_info, now)
        row_params_json = row.get("params_json") or DEFAULT_PARAMS_JSON
        conn.execute(
            """
            DELETE FROM primitive_values
            WHERE raw_smiles = ?
              AND primitive_name = ?
              AND COALESCE(params_json, '{}') = ?
            """,
            [row["raw_smiles"], row["primitive_name"], row_params_json],
        )
        conn.execute(
            """
            INSERT INTO primitive_values(
                raw_smiles, canonical_smiles, raw_is_canonical,
                canonicalization_status, canonicalization_error,
                primitive_name, params_json, value_type, value_json,
                status, error, runtime_name, method, provenance_json, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::JSON, ?, ?, ?, ?, ?::JSON, ?)
            """,
            [
                row["raw_smiles"],
                row.get("canonical_smiles"),
                row.get("raw_is_canonical"),
                row.get("canonicalization_status") or "ok",
                row.get("canonicalization_error"),
                row["primitive_name"],
                row_params_json,
                row.get("value_type") or value_type(row.get("value")),
                row.get("value_json") or encode_value(row.get("value")),
                row.get("status") or "ok",
                row.get("error"),
                row.get("runtime_name"),
                row.get("method"),
                row.get("provenance_json") or encode_value(row.get("provenance") or {}),
                row.get("updated_at_utc") or now,
            ],
        )


def lookup_primitives(
    conn: Any,
    smiles_info: dict[str, Any],
    primitive_names: Iterable[str],
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    row_params_json = params_json(params)
    values: dict[str, Any] = {}
    missing: list[str] = []
    for primitive_name in primitive_names:
        hit = lookup_primitive(conn, smiles_info, primitive_name, row_params_json)
        if hit is None:
            missing.append(primitive_name)
        else:
            values[primitive_name] = hit
    return values, missing


def lookup_primitive(conn: Any, smiles_info: dict[str, Any], primitive_name: str, row_params_json: str) -> Any | None:
    candidates = [smiles_info["raw_smiles"]]
    canonical = smiles_info.get("canonical_smiles")
    if canonical and canonical not in candidates:
        candidates.append(str(canonical))
    for candidate in candidates:
        row = conn.execute(
            """
            SELECT value_json
            FROM primitive_values
            WHERE status = 'ok'
              AND primitive_name = ?
              AND COALESCE(params_json, '{}') = ?
              AND raw_smiles = ?
            ORDER BY updated_at_utc DESC
            LIMIT 1
            """,
            [primitive_name, row_params_json, candidate],
        ).fetchone()
        if row is not None:
            return decode_value(row[0])
        row = conn.execute(
            """
            SELECT value_json
            FROM primitive_values
            WHERE status = 'ok'
              AND primitive_name = ?
              AND COALESCE(params_json, '{}') = ?
              AND canonical_smiles = ?
            ORDER BY updated_at_utc DESC
            LIMIT 1
            """,
            [primitive_name, row_params_json, candidate],
        ).fetchone()
        if row is not None:
            return decode_value(row[0])
    return None


def lookup_bundle_string(conn: Any, smiles_info: dict[str, Any], bundle_name: str) -> str | None:
    for column, value in (
        ("raw_smiles", smiles_info["raw_smiles"]),
        ("canonical_smiles", smiles_info.get("canonical_smiles")),
    ):
        if not value:
            continue
        row = conn.execute(
            f"""
            SELECT text
            FROM bundle_strings
            WHERE status = 'ok'
              AND bundle_name = ?
              AND {column} = ?
            ORDER BY updated_at_utc DESC
            LIMIT 1
            """,
            [bundle_name, value],
        ).fetchone()
        if row is not None:
            return str(row[0] or "")
    return None


def store_bundle_string(
    conn: Any,
    smiles_info: dict[str, Any],
    bundle_name: str,
    text: str,
    primitive_names: Iterable[str],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO bundle_strings(
            raw_smiles, canonical_smiles, bundle_name, status,
            text, error, provenance_json, updated_at_utc
        )
        VALUES (?, ?, ?, 'ok', ?, NULL, ?::JSON, ?)
        """,
        [
            smiles_info["raw_smiles"],
            smiles_info.get("canonical_smiles"),
            bundle_name,
            text,
            encode_value({"primitive_names": list(primitive_names)}),
            utc_now(),
        ],
    )


def lookup_full_call(conn: Any, smiles_info: dict[str, Any], bundle_names: tuple[str, ...]) -> str | None:
    key = feature_set_key(bundle_names)
    for column, value in (
        ("raw_smiles", smiles_info["raw_smiles"]),
        ("canonical_smiles", smiles_info.get("canonical_smiles")),
    ):
        if not value:
            continue
        row = conn.execute(
            f"""
            SELECT feature_text
            FROM full_call_strings
            WHERE status = 'ok'
              AND feature_set_key = ?
              AND {column} = ?
            ORDER BY updated_at_utc DESC
            LIMIT 1
            """,
            [key, value],
        ).fetchone()
        if row is not None:
            return str(row[0] or "")
    return None


def store_full_call(conn: Any, smiles_info: dict[str, Any], bundle_names: tuple[str, ...], text: str) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO full_call_strings(
            raw_smiles, canonical_smiles, feature_set_key, status,
            feature_text, error, provenance_json, updated_at_utc
        )
        VALUES (?, ?, ?, 'ok', ?, NULL, ?::JSON, ?)
        """,
        [
            smiles_info["raw_smiles"],
            smiles_info.get("canonical_smiles"),
            feature_set_key(bundle_names),
            text,
            encode_value({"bundle_names": list(bundle_names)}),
            utc_now(),
        ],
    )
