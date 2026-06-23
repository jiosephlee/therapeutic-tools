# therapeutic-tools

Self-contained therapeutic chemistry tools and local feature caches.

## Install

From this directory:

```bash
python -m pip install -e .
```

This makes `import therapeutic_tools` work without relying on the parent
repository being on `PYTHONPATH`.

## Local Caches

Runtime caches live under `cache/` by default. The standard DuckDB feature cache
is:

```text
cache/therapeutic_tools.duckdb
```

Large database files are local artifacts and are intentionally not tracked by
git. `cache/cache_manifest.json` records the required artifact names and minimum
sizes. Use:

```bash
python -m therapeutic_tools.doctor
```

to verify the local cache, DuckDB importability, and configured runtime
availability.

## Configuration

Useful overrides:

- `THERAPEUTIC_TOOLS_CACHE_DIR`: base cache directory; defaults to
  `therapeutic_tools/cache`.
- `THERAPEUTIC_TOOLS_DUCKDB_CACHE`: explicit DuckDB cache path.
- `THERAPEUTIC_TOOLS_TDC_DATA_DIR`: raw TDC split directory for cache builders.
- `THERAPEUTIC_TOOLS_TDC_DEDUP_DATA_DIR`: deduplicated TDC split directory for
  fingerprint cache building.
- `TRIM_ROOT`: optional external TRIM checkout for TRIM-only tools.
- `ATTNSOM_CHECKPOINT`: optional ATTNSOM checkpoint for live inference.
- `THERAPEUTIC_TOOLS_MINIMOL_REPO`: optional MiniMol source checkout if the
  `minimol` package is not installed in the active environment.

The cache-building/runtime subprocesses still use local conda env paths by
default, notably `THERAPEUTIC_TOOLS_OPENRLHF_PYTHON` and
`THERAPEUTIC_TOOLS_MINIMOL_PYTHON`. Those paths are intentionally left as local
cluster configuration.
