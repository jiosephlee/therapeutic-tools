"""Environment and local-artifact checks for therapeutic_tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from . import paths
from .runtime_envs import RUNTIME_ENVS, validate_runtime_available


def _load_manifest() -> dict[str, Any]:
    manifest_path = paths.PACKAGE_DIR / "cache" / "cache_manifest.json"
    if not manifest_path.exists():
        return {"schema_version": 1, "artifacts": []}
    return json.loads(manifest_path.read_text())


def _artifact_status(entry: dict[str, Any]) -> dict[str, Any]:
    rel = Path(str(entry["path"]))
    path = paths.cache_path(*rel.parts)
    size = path.stat().st_size if path.exists() and path.is_file() else 0
    min_bytes = int(entry.get("min_bytes") or 0)
    ok = path.exists() and (not min_bytes or size >= min_bytes)
    return {
        "path": str(path),
        "required": bool(entry.get("required")),
        "exists": path.exists(),
        "size": size,
        "min_bytes": min_bytes,
        "ok": ok,
    }


def collect_status() -> dict[str, Any]:
    manifest = _load_manifest()
    artifacts = [_artifact_status(entry) for entry in manifest.get("artifacts", [])]
    missing_required = [item for item in artifacts if item["required"] and not item["ok"]]
    runtimes = {name: validate_runtime_available(name) for name in RUNTIME_ENVS}
    return {
        "package_dir": str(paths.PACKAGE_DIR),
        "cache_dir": str(paths.cache_dir()),
        "duckdb_path": str(paths.duckdb_path()),
        "duckdb_module": importlib.util.find_spec("duckdb") is not None,
        "artifacts": artifacts,
        "runtimes": runtimes,
        "ok": not missing_required,
    }


def main() -> int:
    status = collect_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
