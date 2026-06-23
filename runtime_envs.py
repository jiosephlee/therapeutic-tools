"""Runtime registry for model-specific therapeutic-tools endpoints."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPENRLHF_PYTHON = Path(
    os.environ.get(
        "THERAPEUTIC_TOOLS_OPENRLHF_PYTHON",
        "/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python",
    )
)
MINIMOL_PYTHON = Path(
    os.environ.get(
        "THERAPEUTIC_TOOLS_MINIMOL_PYTHON",
        "/vast/projects/myatskar/design-documents/conda_env/minimol/bin/python",
    )
)


@dataclass(frozen=True)
class RuntimeEnv:
    name: str
    python: Path
    required_modules: tuple[str, ...]
    description: str


RUNTIME_ENVS: dict[str, RuntimeEnv] = {
    "openrlhf": RuntimeEnv(
        name="openrlhf",
        python=OPENRLHF_PYTHON,
        required_modules=("rdkit", "molgpka"),
        description="OpenRLHF env for RDKit descriptors, topology, alerts, MolGpKa pKa, ionization, and logD.",
    ),
    "minimol": RuntimeEnv(
        name="minimol",
        python=MINIMOL_PYTHON,
        required_modules=("rdkit", "minimol", "graphium", "torch_sparse"),
        description="MiniMol env for MiniMol + Ridge aqueous solubility.",
    ),
}


ENDPOINT_RUNTIME: dict[str, str] = {
    "v17_molecular_profile_text": "openrlhf",
    "v17_pka_ionization_logd_text": "openrlhf",
    "v17_structure_and_topology_text": "openrlhf",
    "v17_alert_screening_text": "openrlhf",
    "minimol_solubility_log_s": "minimol",
    "esol_solubility_log_s": "openrlhf",
}


class RuntimeMismatchError(RuntimeError):
    """Raised when an endpoint is executed in the wrong Python runtime."""


def get_runtime(name: str) -> RuntimeEnv:
    try:
        return RUNTIME_ENVS[name]
    except KeyError as exc:
        raise KeyError(f"unknown therapeutic-tools runtime {name!r}; valid={sorted(RUNTIME_ENVS)}") from exc


def endpoint_runtime(endpoint_name: str) -> RuntimeEnv:
    try:
        return get_runtime(ENDPOINT_RUNTIME[endpoint_name])
    except KeyError as exc:
        raise KeyError(
            f"unknown therapeutic-tools endpoint {endpoint_name!r}; valid={sorted(ENDPOINT_RUNTIME)}"
        ) from exc


def current_runtime_name() -> str | None:
    exe = Path(sys.executable).resolve()
    for runtime in RUNTIME_ENVS.values():
        try:
            if runtime.python.exists() and exe == runtime.python.resolve():
                return runtime.name
        except OSError:
            continue
    return None


def validate_runtime_available(runtime_name: str) -> dict[str, Any]:
    runtime = get_runtime(runtime_name)
    missing_modules: list[str] = []
    probe_error = None
    if runtime.python.exists():
        code = (
            "import importlib.util, json; "
            f"mods={list(runtime.required_modules)!r}; "
            "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
        )
        try:
            proc = subprocess.run(
                [str(runtime.python), "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                missing_modules = json.loads(proc.stdout.strip() or "[]")
            else:
                probe_error = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
                missing_modules = list(runtime.required_modules)
        except Exception as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
            missing_modules = list(runtime.required_modules)
    else:
        missing_modules = list(runtime.required_modules)
    return {
        "runtime": runtime.name,
        "python": str(runtime.python),
        "python_exists": runtime.python.exists(),
        "current_python": sys.executable,
        "current_runtime": current_runtime_name(),
        "required_modules": list(runtime.required_modules),
        "missing_modules": missing_modules,
        "probe_error": probe_error,
        "ok": runtime.python.exists() and not missing_modules,
    }


def require_runtime_for_endpoint(endpoint_name: str) -> None:
    runtime = endpoint_runtime(endpoint_name)
    status = validate_runtime_available(runtime.name)
    if not status["ok"]:
        raise RuntimeMismatchError(f"runtime {runtime.name!r} is not available: {status}")
    current = current_runtime_name()
    if current != runtime.name:
        raise RuntimeMismatchError(
            f"endpoint {endpoint_name!r} requires runtime {runtime.name!r} "
            f"({runtime.python}), current runtime is {current!r} ({sys.executable})"
        )
