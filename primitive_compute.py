"""Runtime-aware primitive computation for therapeutic features."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import duckdb_store as store
from .primitive_registry import (
    ALERT_PRIMITIVES,
    DEFAULT_PARAMS_JSON,
    MINIMOL_METHOD,
    MOLECULE_PROFILE_PRIMITIVES,
    PKA_PRIMITIVES,
    STRUCTURE_PRIMITIVES,
    compute_families_for_primitives,
    runtime_families,
)
from .runtime_envs import get_runtime, require_runtime_for_endpoint


PACKAGE_DIR = Path(__file__).resolve().parent


def _f_neutral_from_pka(most_acidic: float | None, most_basic: float | None, ph: float = 7.4) -> float:
    """Henderson-Hasselbalch-style neutral fraction from acidic/basic pKa."""
    f_neutral = 1.0
    if most_basic is not None and not (isinstance(most_basic, float) and math.isnan(most_basic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (most_basic - ph))
    if most_acidic is not None and not (isinstance(most_acidic, float) and math.isnan(most_acidic)):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - most_acidic))
    return min(1.0, max(1e-12, f_neutral))


def primitive_row(
    smiles_info: dict[str, Any],
    primitive_name: str,
    value: Any,
    runtime_name: str,
    method: str,
    *,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        **smiles_info,
        "primitive_name": primitive_name,
        "params_json": DEFAULT_PARAMS_JSON,
        "value_type": store.value_type(value),
        "value_json": store.encode_value(value),
        "status": status,
        "error": error,
        "runtime_name": runtime_name,
        "method": method,
        "provenance_json": store.encode_value({"method": method, "runtime_name": runtime_name}),
        "updated_at_utc": store.utc_now(),
    }


def _error_rows(smiles_info: dict[str, Any], primitive_names: Iterable[str], error: str) -> list[dict[str, Any]]:
    from .primitive_registry import PRIMITIVE_SPECS

    rows: list[dict[str, Any]] = []
    for name in primitive_names:
        spec = PRIMITIVE_SPECS[name]
        rows.append(
            primitive_row(
                smiles_info,
                name,
                None,
                spec.runtime_name,
                spec.method,
                status="error",
                error=error,
            )
        )
    return rows


def _compute_rdkit_profile(canonical: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"invalid SMILES: {canonical!r}")
    charges: list[float] = []
    try:
        AllChem.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            try:
                charge = float(atom.GetDoubleProp("_GasteigerCharge"))
            except Exception:
                continue
            if not math.isnan(charge) and not math.isinf(charge):
                charges.append(charge)
    except Exception:
        pass
    atom_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    stereocenters = {
        "r": sum(1 for _, label in atom_centers if label == "R"),
        "s": sum(1 for _, label in atom_centers if label == "S"),
        "unspecified": sum(1 for _, label in atom_centers if label == "?"),
    }
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "HeavyAtomCount": int(Descriptors.HeavyAtomCount(mol)),
        "NumHeteroatoms": int(Lipinski.NumHeteroatoms(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHDonors": int(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": int(Lipinski.NumHAcceptors(mol)),
        "NumRotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "qed": float(QED.qed(mol)),
        "LabuteASA": float(Descriptors.LabuteASA(mol)),
        "NOCount": int(Lipinski.NOCount(mol)),
        "NumAliphaticCarbocycles": int(Lipinski.NumAliphaticCarbocycles(mol)),
        "NumAromaticCarbocycles": int(Lipinski.NumAromaticCarbocycles(mol)),
        "NumSaturatedCarbocycles": int(Lipinski.NumSaturatedCarbocycles(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
        "BalabanJ": float(GraphDescriptors.BalabanJ(mol)),
        "HallKierAlpha": float(GraphDescriptors.HallKierAlpha(mol)),
        "Kappa1": float(GraphDescriptors.Kappa1(mol)),
        "Kappa2": float(GraphDescriptors.Kappa2(mol)),
        "Kappa3": float(GraphDescriptors.Kappa3(mol)),
        "NumAtomStereoCenters": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "NumUnspecifiedAtomStereoCenters": int(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)),
        "stereocenter_breakdown": stereocenters,
        "MaxAbsPartialCharge": max((abs(v) for v in charges), default=None),
        "MinAbsPartialCharge": min((abs(v) for v in charges), default=None),
        "charge_polarization": (max(charges) - min(charges)) if charges else None,
        "NumAromaticAtoms": sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
        "RingCount": int(Lipinski.RingCount(mol)),
        "NumAromaticRings": int(Lipinski.NumAromaticRings(mol)),
        "NumAliphaticRings": int(Lipinski.NumAliphaticRings(mol)),
        "NumSaturatedRings": int(Lipinski.NumSaturatedRings(mol)),
        "NumHeterocycles": int(Lipinski.NumHeterocycles(mol)),
        "neutral_fraction_7_4": _f_neutral_from_pka(None, None, 7.4),
    }


def _uncached_compact_ionization(smiles: str, ph: float = 7.4) -> str:
    from collections import Counter

    from rdkit import Chem

    try:
        from therapeutic_tools.utils.legacy_tools.RDKit_tools import protonate_smiles
    except Exception:
        protonate_smiles = None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        variants = protonate_smiles(canonical, ph_min=ph, ph_max=ph, precision=0.5) if protonate_smiles else []
    except Exception:
        variants = []
    variant_data = []
    for variant in variants or [canonical]:
        vmol = Chem.MolFromSmiles(variant)
        if vmol is None:
            continue
        pos = neg = 0
        for atom in vmol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                pos += charge
            elif charge < 0:
                neg += abs(charge)
        net = pos - neg
        charge_class = "zwitterion" if pos and neg else "base" if pos else "acid" if neg else "neutral"
        variant_data.append({"smiles": variant, "net_charge": net, "charge_class": charge_class})
    if not variant_data:
        variant_data = [{"smiles": canonical, "net_charge": 0, "charge_class": "neutral"}]
    charges = sorted({row["net_charge"] for row in variant_data})
    counts = Counter(row["net_charge"] for row in variant_data)
    mode_charge = max(counts, key=lambda charge: (counts[charge], -abs(charge)))
    representative = next(row for row in variant_data if row["net_charge"] == mode_charge)
    return "\n".join(
        [
            f"Ionization at pH {ph}: {representative['charge_class']}, charge {representative['net_charge']}",
            f"- Dominant form: {representative['smiles']}",
            f"- Ambiguous (pKa near pH): {'Yes' if len(charges) > 1 else 'No'}",
        ]
    )


def _compute_pka_ionization_logd(canonical: str) -> dict[str, Any]:
    from rdkit.Chem import Crippen

    from therapeutic_tools.utils.adme import _estimate_logd_from_pka, _format_pka_section, _get_pka_data

    pka_data = _get_pka_data(canonical, None)
    ionization_text = _uncached_compact_ionization(canonical, ph=7.4)
    logd_74 = float(_estimate_logd_from_pka(canonical, 7.4, pka_data, None))
    if math.isnan(logd_74) or math.isinf(logd_74):
        from rdkit import Chem

        logd_74 = float(Crippen.MolLogP(Chem.MolFromSmiles(canonical)))
    text = "\n".join(
        [
            _format_pka_section(canonical, pka_data, simple=False),
            "",
            ionization_text,
            "",
            f"LogD at pH 7.4: {logd_74:.2f}",
        ]
    )
    return {"pka": pka_data, "ionization_text": ionization_text, "logD_74": logd_74, "text": text}


def _compute_structure_values(canonical: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    from therapeutic_tools.utils.endpoints import functional_group_names

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"invalid SMILES: {canonical!r}")
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    return {
        "functional_groups": functional_group_names(canonical) or [],
        "ring_count": int(Lipinski.RingCount(mol)),
        "aromatic_rings": int(Lipinski.NumAromaticRings(mol)),
        "aliphatic_rings": int(Lipinski.NumAliphaticRings(mol)),
        "saturated_rings": int(Lipinski.NumSaturatedRings(mol)),
        "heterocycles": int(Lipinski.NumHeterocycles(mol)),
        "largest_ring_size": max((len(ring) for ring in atom_rings), default=0),
    }


def _compute_alert_values(canonical: str) -> dict[str, Any]:
    from collections import OrderedDict

    from therapeutic_tools.utils.safety import _screen_structural_alerts_data, _screen_toxalerts_data

    merged: OrderedDict[str, Any] = OrderedDict()
    for data in (_screen_structural_alerts_data(canonical, include_smarts=False), _screen_toxalerts_data(canonical, include_smarts=False)):
        for category, value in data.items():
            if category not in merged:
                merged[category] = {"note": value["note"], "alerts": list(value["alerts"])}
                continue
            seen = {str(alert).lower() for alert in merged[category]["alerts"]}
            for alert in value["alerts"]:
                if str(alert).lower() not in seen:
                    merged[category]["alerts"].append(alert)
                    seen.add(str(alert).lower())
    return {"alert_categories": list(merged), "alert_category_count": len(merged)}


def _compute_minimol_log_s(canonical: str) -> float:
    require_runtime_for_endpoint("minimol_solubility_log_s")
    predictor = _load_minimol_predictor()
    pred = predictor.predict([canonical])
    value = float(pred[0] if isinstance(pred, list) else pred)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"invalid MiniMol prediction {pred!r}")
    return value


def compute_primitives_local(smiles: str, primitive_names: Iterable[str]) -> list[dict[str, Any]]:
    """Compute primitive rows in the current process without touching DuckDB."""
    from .primitive_registry import validate_primitives

    names = validate_primitives(primitive_names)
    smiles_info = store.canonicalize_smiles(smiles)
    if smiles_info["canonicalization_status"] != "ok":
        return _error_rows(smiles_info, names, str(smiles_info["canonicalization_error"]))

    canonical = str(smiles_info["canonical_smiles"])
    rows: list[dict[str, Any]] = []
    families = compute_families_for_primitives(names)
    if "molecule_profile" in families:
        values = _compute_rdkit_profile(canonical)
        for name in families["molecule_profile"]:
            rows.append(primitive_row(smiles_info, name, values.get(name), "openrlhf", "RDKit descriptors"))
    if "pka_ionization_logd" in families:
        values = _compute_pka_ionization_logd(canonical)
        for name in families["pka_ionization_logd"]:
            key = "pka" if name == "pka_summary" else "text" if name == "pka_ionization_logd_text" else name
            rows.append(primitive_row(smiles_info, name, values.get(key), "openrlhf", "MolGpKa/RDKit"))
    if "structure_and_topology" in families:
        values = _compute_structure_values(canonical)
        for name in families["structure_and_topology"]:
            rows.append(primitive_row(smiles_info, name, values.get(name), "openrlhf", "RDKit SMARTS/ring topology"))
    if "alert_screening" in families:
        values = _compute_alert_values(canonical)
        for name in families["alert_screening"]:
            rows.append(primitive_row(smiles_info, name, values.get(name), "openrlhf", "RDKit structural alerts"))
    if "minimol_solubility" in families:
        value = _compute_minimol_log_s(canonical)
        for name in families["minimol_solubility"]:
            rows.append(primitive_row(smiles_info, name, value, "minimol", MINIMOL_METHOD))
    return rows


def compute_missing_primitives(smiles: str, primitive_names: Iterable[str]) -> list[dict[str, Any]]:
    """Compute missing primitive rows via configured runtime subprocesses."""
    from .primitive_registry import validate_primitives

    names = validate_primitives(primitive_names)
    rows: list[dict[str, Any]] = []
    families = compute_families_for_primitives(names)
    family_to_names = {family: tuple(items) for family, items in families.items()}
    for runtime_name, runtime_family_names in runtime_families(families).items():
        runtime = get_runtime(runtime_name)
        if not runtime.python.exists():
            raise RuntimeError(f"runtime {runtime_name!r} Python does not exist: {runtime.python}")
        primitive_batch = [
            primitive
            for family in runtime_family_names
            for primitive in family_to_names[family]
        ]
        cmd = [
            str(runtime.python),
            "-m",
            "therapeutic_tools.primitive_compute",
            "compute-primitives",
            "--smiles",
            smiles,
            "--primitive-names",
            *primitive_batch,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PACKAGE_DIR.parent)
        env["PYTHONNOUSERSITE"] = "1"
        env["THERAPEUTIC_TOOLS_DISABLE_DUCKDB_CACHE"] = "1"
        env["THERAPEUTIC_TOOLS_DISABLE_STRUCTURED_FEATURE_CACHE"] = "1"
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{runtime_name} primitive worker failed with exit code {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        payload = json.loads(proc.stdout)
        rows.extend(payload["primitive_rows"])
    return rows


def compute_minimol_records(smiles_list: list[str], batch_size: int = 64) -> Iterator[dict[str, Any]]:
    """Compatibility endpoint-record generator for runtime_cache shard builds."""
    require_runtime_for_endpoint("minimol_solubility_log_s")
    predictor = _load_minimol_predictor()
    for start in range(0, len(smiles_list), batch_size):
        raw_batch = smiles_list[start : start + batch_size]
        infos = [store.canonicalize_smiles(smiles) for smiles in raw_batch]
        valid_infos = [info for info in infos if info["canonicalization_status"] == "ok"]
        valid_smiles = [str(info["canonical_smiles"]) for info in valid_infos]
        predictions: list[Any] = []
        if valid_smiles:
            try:
                pred = predictor.predict(valid_smiles)
                predictions = list(pred if isinstance(pred, list) else [pred])
            except Exception as exc:
                predictions = [exc] * len(valid_smiles)
        pred_by_raw = {info["raw_smiles"]: pred for info, pred in zip(valid_infos, predictions)}
        for info in infos:
            if info["canonicalization_status"] != "ok":
                yield {
                    **info,
                    "endpoint_name": "minimol_solubility_log_s",
                    "endpoint_version": "v1",
                    "runtime_name": "minimol",
                    "python_executable": sys.executable,
                    "status": "error",
                    "value_json": None,
                    "text": None,
                    "error": info["canonicalization_error"],
                    "updated_at_utc": store.utc_now(),
                }
                continue
            pred = pred_by_raw.get(info["raw_smiles"])
            if isinstance(pred, Exception):
                yield {
                    **info,
                    "endpoint_name": "minimol_solubility_log_s",
                    "endpoint_version": "v1",
                    "runtime_name": "minimol",
                    "python_executable": sys.executable,
                    "status": "error",
                    "value_json": None,
                    "text": None,
                    "error": f"{type(pred).__name__}: {pred}",
                    "updated_at_utc": store.utc_now(),
                }
                continue
            value = float(pred)
            yield {
                **info,
                "endpoint_name": "minimol_solubility_log_s",
                "endpoint_version": "v1",
                "runtime_name": "minimol",
                "python_executable": sys.executable,
                "status": "ok",
                "value_json": {"log_s": value, "method": MINIMOL_METHOD},
                "text": f"Solubility: logS = {value:.2f} log(mol/L)",
                "error": None,
                "updated_at_utc": store.utc_now(),
            }


def _load_minimol_predictor():
    module_path = PACKAGE_DIR.parent / "utils" / "minimol_solubility.py"
    spec = importlib.util.spec_from_file_location("therapeutic_tuning_minimol_solubility", module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"could not load MiniMolSolubility from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniMolSolubility()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_compute = sub.add_parser("compute-primitives")
    p_compute.add_argument("--smiles", required=True)
    p_compute.add_argument("--primitive-names", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "compute-primitives":
        with contextlib.redirect_stdout(sys.stderr):
            rows = compute_primitives_local(args.smiles, args.primitive_names)
        print(json.dumps({"primitive_rows": rows}, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
