"""Public DuckDB-first facade for therapeutic feature primitives and bundles."""

from __future__ import annotations

from typing import Any, Iterable

from . import duckdb_store as store
from . import primitive_compute, primitive_registry, renderers


def get_primitives(
    smiles: str,
    primitive_names: Iterable[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return typed primitive values, computing and writing through on misses."""
    if store.duckdb_disabled():
        raise RuntimeError("DuckDB cache is disabled; primitive facade requires DuckDB")
    names = primitive_registry.validate_primitives(primitive_names)
    smiles_info = store.canonicalize_smiles(smiles)
    if smiles_info["canonicalization_status"] != "ok":
        raise ValueError(smiles_info["canonicalization_error"])

    with store.connect() as conn:
        store.upsert_molecule(conn, smiles_info)
        values, missing = store.lookup_primitives(conn, smiles_info, names, params)
        if missing:
            rows = primitive_compute.compute_missing_primitives(smiles, missing)
            store.write_primitive_rows(conn, rows)
            loaded, still_missing = store.lookup_primitives(conn, smiles_info, missing, params)
            if still_missing:
                raise RuntimeError(
                    "primitive row(s) were not written for "
                    f"{smiles!r}: {', '.join(still_missing)}"
                )
            values.update(loaded)
        return {name: values[name] for name in names}


def get_primitive(smiles: str, primitive_name: str, params: dict[str, Any] | None = None) -> Any:
    """Convenience wrapper around :func:`get_primitives` for one value."""
    return get_primitives(smiles, [primitive_name], params)[primitive_name]


def get_bundle(smiles: str, bundle_name: str) -> str:
    """Return a named semantic bundle rendered from primitive rows."""
    if store.duckdb_disabled():
        raise RuntimeError("DuckDB cache is disabled; bundle facade requires DuckDB")
    normalized = primitive_registry.normalize_bundle_name(bundle_name)
    if normalized == "all_features":
        return get_all_features(smiles)
    primitive_names = primitive_registry.primitives_for_bundle(normalized)
    smiles_info = store.canonicalize_smiles(smiles)
    if smiles_info["canonicalization_status"] != "ok":
        raise ValueError(smiles_info["canonicalization_error"])

    with store.connect() as conn:
        cached = store.lookup_bundle_string(conn, smiles_info, normalized)
        if cached is not None:
            return cached
    values = get_primitives(smiles, primitive_names)
    text = renderers.render_bundle(normalized, values)
    with store.connect() as conn:
        store.store_bundle_string(conn, smiles_info, normalized, text, primitive_names)
    return text


def get_all_features(smiles: str) -> str:
    """Return all feature bundles in the public v17 order."""
    return get_feature_text(smiles, primitive_registry.PUBLIC_FEATURE_NAMES)


def get_feature_text(smiles: str, feature_names: Iterable[str]) -> str:
    """Return ordered feature text for v17-style feature group names."""
    if store.duckdb_disabled():
        raise RuntimeError("DuckDB cache is disabled; feature facade requires DuckDB")
    resolved, errors = primitive_registry.resolve_feature_names(feature_names)
    if errors:
        raise KeyError(
            "unknown therapeutic bundle(s): "
            f"{', '.join(errors)}"
        )
    bundle_names = tuple(primitive_registry.normalize_bundle_name(name) for name in resolved)
    smiles_info = store.canonicalize_smiles(smiles)
    if smiles_info["canonicalization_status"] != "ok":
        raise ValueError(smiles_info["canonicalization_error"])

    with store.connect() as conn:
        cached = store.lookup_full_call(conn, smiles_info, bundle_names)
        if cached is not None:
            return cached
    sections = [get_bundle(smiles, name) for name in bundle_names]
    text = "\n\n".join(section for section in sections if section)
    with store.connect() as conn:
        store.store_full_call(conn, smiles_info, bundle_names, text)
    return text


__all__ = [
    "get_all_features",
    "get_bundle",
    "get_feature_text",
    "get_primitive",
    "get_primitives",
]

