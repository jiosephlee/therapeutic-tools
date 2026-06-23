"""Package-local path resolution for therapeutic_tools."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PACKAGE_DIR / "cache"
DEFAULT_TDC_DATA_DIR = DEFAULT_CACHE_DIR / "tdc" / "raw"


def env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


def cache_dir() -> Path:
    return env_path("THERAPEUTIC_TOOLS_CACHE_DIR", DEFAULT_CACHE_DIR) or DEFAULT_CACHE_DIR


def cache_path(*parts: str) -> Path:
    return cache_dir().joinpath(*parts)


def tdc_data_dir() -> Path:
    return env_path("THERAPEUTIC_TOOLS_TDC_DATA_DIR", cache_path("tdc", "raw")) or DEFAULT_TDC_DATA_DIR


def duckdb_path() -> Path:
    return (
        env_path("THERAPEUTIC_TOOLS_DUCKDB_CACHE")
        or env_path("THERAPEUTIC_TOOLS_RUNTIME_CACHE_DUCKDB")
        or env_path("THERAPEUTIC_TOOLS_FEATURE_CACHE_DUCKDB")
        or cache_path("therapeutic_tools.duckdb")
    )


def structured_feature_cache_path() -> Path:
    return (
        env_path("THERAPEUTIC_TOOLS_STRUCTURED_FEATURE_CACHE")
        or cache_path("structured_feature_cache.sqlite")
    )


def optional_path(name: str, *default_parts: str) -> Path | None:
    configured = env_path(name)
    if configured is not None:
        return configured
    candidate = cache_path(*default_parts)
    return candidate if candidate.exists() else None


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_TDC_DATA_DIR",
    "PACKAGE_DIR",
    "cache_dir",
    "cache_path",
    "duckdb_path",
    "env_path",
    "optional_path",
    "structured_feature_cache_path",
    "tdc_data_dir",
]
