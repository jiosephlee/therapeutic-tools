"""
Phase 3: Train TRIM EBM models and regenerate manifests.

Calls TRIM's own training APIs to produce:
  - outputs/models/global_ebm/{experiment}/{task}/{feature_set}/model_bundle.pkl
  - outputs/models/pair_ebm/{experiment}/{task}/{feature_set}/pos_model_bundle.pkl
  - outputs/models/pair_ebm/{experiment}/{task}/{feature_set}/neg_model_bundle.pkl
  - outputs/reasoning_agent_tools/manifests/{feature_set}/{task}.json

Usage:
    TRIM_PROJECT_ROOT=/path/to/trim_artifacts \\
    python train_trim_models.py --trim-root /path/to/trim_artifacts

    # Single task for testing:
    python train_trim_models.py --trim-root /path/to/trim_artifacts --tasks BBB_Martins
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TRIM_ROOT = _REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "TRIM"

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma", "Carcinogens_Lagunin",
    "ClinTox", "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "DILI", "hERG", "HIA_Hou",
    "PAMPA_NCATS", "Pgp_Broccatelli", "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret", "Skin_Reaction",
]

# Must match the config filename stem in TRIM/configs/features/
DEFAULT_FEATURE_CONFIG = "fg_top_level_plus_rdkit_descriptors_and_pka_easy_to_NLP_Lv1_core_pka_no_fr_counts"
DEFAULT_EXPERIMENT_NAME = "retrained_v1"


def setup_trim_path(trim_root: Path) -> None:
    """Add TRIM/src to sys.path and set TRIM_PROJECT_ROOT."""
    trim_src = _DEFAULT_TRIM_ROOT / "src"
    if str(trim_src) not in sys.path:
        sys.path.insert(0, str(trim_src))
    os.environ["TRIM_PROJECT_ROOT"] = str(trim_root)


def train_global(
    task: str,
    trim_root: Path,
    feature_config: str,
    experiment_name: str,
) -> dict:
    """Train global EBM for a single task."""
    from trim.features.table_loader import build_feature_source_bundle
    from trim.training.global_training import train_global_task

    config_path = _DEFAULT_TRIM_ROOT / "configs" / "features" / f"{feature_config}.json"
    processed_root = trim_root / "data" / "processed" / "tdc_no_conflict_labels_salt_removed"
    output_dir = trim_root / "outputs" / "models" / "global_ebm" / experiment_name

    log.info("[global] %s: building features from %s", task, config_path.name)
    feature_bundle = build_feature_source_bundle([str(config_path)])

    log.info("[global] %s: training ...", task)
    summary = train_global_task(
        task=task,
        feature_bundle=feature_bundle,
        dataset_root=str(processed_root),
        train_split_name="train",
        valid_split_name="valid",
        smiles_key="drug",
        label_key="Y",
        output_dir=str(output_dir),
    )

    metrics = summary.get("final_model_metrics", {})
    train_auc = metrics.get("train", {}).get("auroc")
    valid_auc = metrics.get("valid", {}).get("auroc")
    log.info("[global] %s: train_auroc=%s  valid_auroc=%s", task, train_auc, valid_auc)
    return summary


def train_pair(
    task: str,
    trim_root: Path,
    feature_config: str,
    experiment_name: str,
    neighbor_label: int,
    n_jobs: int = 1,
) -> dict:
    """Train pair EBM (pos or neg) for a single task."""
    from trim.features.table_loader import build_feature_source_bundle
    from trim.models.retrieval import CachedSimilarityRetriever
    from trim.training.pair_training import PairTrainingConfig, train_pair_task

    config_path = _DEFAULT_TRIM_ROOT / "configs" / "features" / f"{feature_config}.json"
    processed_root = trim_root / "data" / "processed" / "tdc_no_conflict_labels_salt_removed"
    cache_root = trim_root / "data" / "cache" / "tdc_mol_fingerprints"
    output_dir = trim_root / "outputs" / "models" / "pair_ebm" / experiment_name

    pair_name = "pos" if neighbor_label == 1 else "neg"
    log.info("[pair-%s] %s: building features ...", pair_name, task)
    feature_bundle = build_feature_source_bundle([str(config_path)])

    log.info("[pair-%s] %s: setting up retriever (cache_root=%s) ...", pair_name, task, cache_root)
    retriever = CachedSimilarityRetriever(
        cache_root=str(cache_root),
        data_root=str(processed_root),
    )

    config = PairTrainingConfig(
        neighbor_label=neighbor_label,
        top_k=3,
        strict_cross_scaffold_pairs=True,
        n_jobs=n_jobs,
    )

    log.info("[pair-%s] %s: training ...", pair_name, task)
    summary = train_pair_task(
        task=task,
        feature_bundle=feature_bundle,
        retriever=retriever,
        config=config,
        output_dir=str(output_dir),
    )

    metrics = summary.get("metrics", {})
    train_auc = metrics.get("train", {}).get("auroc")
    valid_auc = metrics.get("valid", {}).get("auroc")
    log.info("[pair-%s] %s: train_auroc=%s  valid_auroc=%s", pair_name, task, train_auc, valid_auc)
    return summary


def regenerate_manifests(
    trim_root: Path,
    tasks: list[str],
    feature_config: str,
) -> None:
    """Regenerate JSON manifests with correct local paths."""
    from trim.reasoning.agent_tools.manifests import build_all_task_tool_manifests

    processed_root = trim_root / "data" / "processed" / "tdc_no_conflict_labels_salt_removed"
    cache_root = trim_root / "data" / "cache" / "tdc_mol_fingerprints"
    outputs_root = trim_root / "outputs"
    manifest_root = outputs_root / "reasoning_agent_tools" / "manifests"

    # The feature_set_name is inferred from the config; use the same one
    # that DEFAULT_AGENT_TOOL_FEATURE_SET_NAME expects.
    feature_set_name = feature_config.replace(
        "fg_top_level_plus_", "fg_top_level+"
    )

    log.info("Regenerating manifests for %d tasks (feature_set=%s) ...", len(tasks), feature_set_name)
    summary = build_all_task_tool_manifests(
        tasks=tasks,
        feature_set_name=feature_set_name,
        dataset_root=str(processed_root),
        cache_root=str(cache_root),
        outputs_root=str(outputs_root),
        manifest_root=str(manifest_root),
    )
    log.info("Manifests written: %d tasks", summary.get("num_tasks", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRIM EBM models and generate manifests.")
    parser.add_argument("--trim-root", type=str, default=str(_DEFAULT_TRIM_ROOT))
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--feature-config", type=str, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for pair EBM fitting")
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--skip-pair", action="store_true")
    parser.add_argument("--skip-manifests", action="store_true")
    args = parser.parse_args()

    trim_root = Path(args.trim_root).resolve()
    tasks = args.tasks or TASKS

    setup_trim_path(trim_root)

    # --- Global EBMs ---
    if not args.skip_global:
        log.info("=== Training Global EBMs (%d tasks) ===", len(tasks))
        for task in tasks:
            try:
                train_global(task, trim_root, args.feature_config, args.experiment_name)
            except Exception as exc:
                log.error("[global] %s FAILED: %s", task, exc, exc_info=True)

    # --- Pair EBMs ---
    if not args.skip_pair:
        log.info("=== Training Pair EBMs (%d tasks × 2) ===", len(tasks))
        for task in tasks:
            for neighbor_label in (1, 0):
                pair_name = "pos" if neighbor_label == 1 else "neg"
                try:
                    train_pair(task, trim_root, args.feature_config, args.experiment_name,
                               neighbor_label=neighbor_label, n_jobs=args.n_jobs)
                except Exception as exc:
                    log.error("[pair-%s] %s FAILED: %s", pair_name, task, exc, exc_info=True)

    # --- Manifests ---
    if not args.skip_manifests:
        log.info("=== Regenerating Manifests ===")
        try:
            regenerate_manifests(trim_root, tasks, args.feature_config)
        except Exception as exc:
            log.error("Manifest generation FAILED: %s", exc, exc_info=True)

    log.info("All done.")


if __name__ == "__main__":
    main()
