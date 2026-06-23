"""Persistent cache for typed low-level therapeutic feature values.

High-level semantic tools should compose values from ``utils.endpoints`` and
format them as text. This cache stores the typed endpoint outputs before
formatting, keyed by endpoint name, SMILES, and endpoint parameters.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .. import paths

PACKAGE_DIR = paths.PACKAGE_DIR
DEFAULT_CACHE_PATH = paths.cache_path("structured_feature_cache.sqlite")
DEFAULT_CACHE_VERSION = "v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enabled() -> bool:
    if os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND", "").lower() == "duckdb_only":
        return False
    value = os.environ.get("THERAPEUTIC_TOOLS_DISABLE_STRUCTURED_FEATURE_CACHE", "")
    return value.lower() not in {"1", "true", "yes", "on"}


def _cache_path() -> Path:
    return paths.structured_feature_cache_path()


def _cache_version() -> str:
    return os.environ.get("THERAPEUTIC_TOOLS_STRUCTURED_FEATURE_CACHE_VERSION", DEFAULT_CACHE_VERSION)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def _params_key(params: dict[str, Any] | None) -> str:
    return json.dumps(_normalize_for_json(params or {}), sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=None)
def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS structured_feature_values (
            endpoint_name TEXT NOT NULL,
            smiles TEXT NOT NULL,
            params_json TEXT NOT NULL,
            cache_version TEXT NOT NULL,
            value_json TEXT,
            status TEXT NOT NULL,
            error TEXT,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(endpoint_name, smiles, params_json, cache_version)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_structured_feature_values_status "
        "ON structured_feature_values(status)"
    )
    conn.commit()
    return conn


def get_or_compute(
    endpoint_name: str,
    smiles: str,
    params: dict[str, Any] | None,
    compute: Callable[[], Any],
) -> Any:
    """Return a cached typed endpoint value, or compute and persist it."""
    if not _enabled():
        return compute()

    params_json = _params_key(params)
    version = _cache_version()
    conn = _connect(str(_cache_path()))
    row = conn.execute(
        """
        SELECT value_json
        FROM structured_feature_values
        WHERE endpoint_name = ?
          AND smiles = ?
          AND params_json = ?
          AND cache_version = ?
          AND status = 'ok'
        """,
        (endpoint_name, smiles, params_json, version),
    ).fetchone()
    if row is not None:
        return json.loads(str(row[0]))

    try:
        value = compute()
        value_json = json.dumps(_normalize_for_json(value), sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        conn.execute(
            """
            INSERT INTO structured_feature_values(
                endpoint_name, smiles, params_json, cache_version,
                value_json, status, error, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(endpoint_name, smiles, params_json, cache_version) DO UPDATE SET
                value_json = excluded.value_json,
                status = excluded.status,
                error = excluded.error,
                updated_at_utc = excluded.updated_at_utc
            """,
            (endpoint_name, smiles, params_json, version, None, "error", f"{type(exc).__name__}: {exc}", _utc_now()),
        )
        conn.commit()
        raise

    conn.execute(
        """
        INSERT INTO structured_feature_values(
            endpoint_name, smiles, params_json, cache_version,
            value_json, status, error, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(endpoint_name, smiles, params_json, cache_version) DO UPDATE SET
            value_json = excluded.value_json,
            status = excluded.status,
            error = excluded.error,
            updated_at_utc = excluded.updated_at_utc
        """,
        (endpoint_name, smiles, params_json, version, value_json, "ok", None, _utc_now()),
    )
    conn.commit()
    return value


def clear_connections() -> None:
    _connect.cache_clear()
