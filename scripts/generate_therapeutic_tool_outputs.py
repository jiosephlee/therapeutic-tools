#!/usr/bin/env python3
"""Generate example output dumps for therapeutic tool versions.

Writes one text file per available versioned therapeutic-tool surface into an
output directory. The default example molecule is aspirin and the default task
context for neighbor-comparison tools is AMES.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
DEFAULT_NAME = "aspirin"
DEFAULT_TASK = "AMES"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "tool_outputs"


@dataclass(frozen=True)
class OutputSpec:
    version: str
    render: Callable[[str, str], str]


def _header(version: str, smiles: str, molecule_name: str, task_name: str) -> list[str]:
    return [
        f"Therapeutic tool outputs for {version}",
        f"Example molecule: {molecule_name}",
        f"SMILES: {smiles}",
        f"Task context: {task_name}",
        "",
    ]


def _render_section(title: str, body: str) -> str:
    return "\n".join(
        [
            f"=== {title} ===",
            body.strip() if body.strip() else "(empty output)",
            "",
        ]
    )


def _render_v10(smiles: str, task_name: str) -> str:
    mod = importlib.import_module("openrlhf.tools.therapeutic_tools.v10")
    parts = [
        _render_section("get_molecular_properties", mod.get_molecular_properties(smiles)),
        _render_section("get_similar_neighbors", mod.get_similar_neighbors(smiles, task_name)),
    ]
    return "\n".join(parts).rstrip() + "\n"


def _render_features_version(module_name: str, smiles: str, task_name: str, include_neighbors: bool) -> str:
    mod = importlib.import_module(module_name)
    feature_names = list(mod.FEATURE_NAMES)
    parts = [
        _render_section(
            "get_features",
            mod.get_features(smiles, feature_names),
        )
    ]
    if include_neighbors:
        parts.append(
            _render_section(
                "get_neighbors",
                mod.get_neighbors(smiles, task_name=task_name, feature_names=feature_names),
            )
        )
    return "\n".join(parts).rstrip() + "\n"


def _render_v11(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v11", smiles, task_name, True)


def _render_v12(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v12", smiles, task_name, True)


def _render_v13(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v13", smiles, task_name, True)


def _render_v14(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v14", smiles, task_name, True)


def _render_v14_no_neighbor(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v14_no_neighbor", smiles, task_name, False)


def _render_v14_consolidated(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v14_consolidated", smiles, task_name, True)


def _render_v14_consolidated_no_neighbor(smiles: str, task_name: str) -> str:
    return _render_features_version(
        "openrlhf.tools.therapeutic_tools.v14_consolidated_no_neighbor", smiles, task_name, False
    )


def _render_v15(smiles: str, task_name: str) -> str:
    mod = importlib.import_module("openrlhf.tools.therapeutic_tools.v15")
    parts = [
        _render_section("get_mol_properties_and_fg", mod.get_mol_properties_and_fg(smiles)),
        _render_section("compare_similar_mols", mod.compare_similar_mols(smiles, task=task_name)),
    ]
    return "\n".join(parts).rstrip() + "\n"


def _render_v16(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v16", smiles, task_name, True)


def _render_v16_no_neighbor(smiles: str, task_name: str) -> str:
    return _render_features_version("openrlhf.tools.therapeutic_tools.v16_no_neighbor", smiles, task_name, False)


OUTPUT_SPECS: list[OutputSpec] = [
    OutputSpec("v10", _render_v10),
    OutputSpec("v11", _render_v11),
    OutputSpec("v12", _render_v12),
    OutputSpec("v13", _render_v13),
    OutputSpec("v14", _render_v14),
    OutputSpec("v14_no_neighbor", _render_v14_no_neighbor),
    OutputSpec("v14_consolidated", _render_v14_consolidated),
    OutputSpec("v14_consolidated_no_neighbor", _render_v14_consolidated_no_neighbor),
    OutputSpec("v15", _render_v15),
    OutputSpec("v16", _render_v16),
    OutputSpec("v16_no_neighbor", _render_v16_no_neighbor),
]


def generate_outputs(
    output_dir: Path,
    smiles: str,
    molecule_name: str,
    task_name: str,
    versions: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_versions = set(versions or [])
    for spec in OUTPUT_SPECS:
        if requested_versions and spec.version not in requested_versions:
            continue
        content = "\n".join(_header(spec.version, smiles, molecule_name, task_name))
        content += spec.render(smiles, task_name)
        out_path = output_dir / f"{spec.version}.txt"
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate example therapeutic tool outputs")
    parser.add_argument("--smiles", default=DEFAULT_SMILES, help="Example molecule as SMILES")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Display name for the example molecule")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task context for neighbor-based tools")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for versioned text outputs")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=[spec.version for spec in OUTPUT_SPECS],
        default=None,
        help="Optional subset of tool versions to render.",
    )
    args = parser.parse_args()

    generate_outputs(
        output_dir=Path(args.output_dir),
        smiles=args.smiles,
        molecule_name=args.name,
        task_name=args.task,
        versions=args.versions,
    )


if __name__ == "__main__":
    main()
