"""Offline runtime-aware DuckDB feature cache builder.

This is not the public feature API. Public callers should use
``therapeutic_tools.api``. This module exists for one-time and batch backfills:
it shards SMILES across the configured runtime Python environments, writes
intermediate endpoint records, and publishes atomic primitive rows plus derived
bundle/full-call strings into DuckDB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import primitive_compute, renderers
from .runtime_envs import get_runtime, require_runtime_for_endpoint


FEATURE_GROUPS = (
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
    "concise_profile",
)
OPENRLHF_ENDPOINTS = (
    "v17_molecular_profile_text",
    "v17_pka_ionization_logd_text",
    "v17_structure_and_topology_text",
    "v17_alert_screening_text",
)
MINIMOL_ENDPOINTS = ("minimol_solubility_log_s",)
BAD_TEXT_MARKERS = (
    "No module named 'molgpka'",
    "locking protocol",
    "database is locked",
    "Traceback",
    ": Error - ",
)
DUCKDB_ONLY_BACKEND = "duckdb_only"


def duckdb_only_mode() -> bool:
    return os.environ.get("THERAPEUTIC_TOOLS_CACHE_BACKEND", "").lower() == DUCKDB_ONLY_BACKEND


def ensure_not_bad_text(endpoint_name: str, text: str | None) -> None:
    reason = bad_text_reason(text)
    if reason is not None:
        raise ValueError(f"{endpoint_name}:{reason}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def value_as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=json_default) + "\n")
            count += 1
    return count


def write_records(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write endpoint records as Parquet when requested, otherwise JSONL."""
    if path.suffix == ".parquet":
        import pandas as pd

        records = list(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = []
        for row in records:
            item = dict(row)
            if not isinstance(item.get("value_json"), str):
                item["value_json"] = json.dumps(item.get("value_json"), sort_keys=True, default=json_default)
            normalized.append(item)
        pd.DataFrame(normalized).to_parquet(path, index=False)
        return len(records)
    return write_jsonl(path, rows)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            yield row
        return
    yield from read_jsonl(path)


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


def bad_text_reason(text: str | None) -> str | None:
    if not text:
        return "empty_text"
    for marker in BAD_TEXT_MARKERS:
        if marker in text:
            return f"bad_text_marker:{marker}"
    return None


def endpoint_record(
    *,
    smiles_info: dict[str, Any],
    endpoint_name: str,
    runtime_name: str,
    status: str,
    value: Any = None,
    text: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        **smiles_info,
        "endpoint_name": endpoint_name,
        "endpoint_version": "v1",
        "runtime_name": runtime_name,
        "python_executable": sys.executable,
        "status": status,
        "value_json": value,
        "text": text,
        "error": error,
        "updated_at_utc": utc_now(),
    }


def _format_float(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}{suffix}"


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
        "neutral_fraction_7_4": primitive_compute._f_neutral_from_pka(None, None, 7.4),
    }


def render_molecular_profile_from_primitives(values: dict[str, Any]) -> str:
    stereo = values.get("stereocenter_breakdown") or {}
    stereo_total = int(values.get("NumAtomStereoCenters") or 0)
    stereo_line = f"- Stereocenters: {stereo_total}"
    if stereo_total:
        parts = []
        if stereo.get("r"):
            parts.append(f"R={stereo['r']}")
        if stereo.get("s"):
            parts.append(f"S={stereo['s']}")
        if stereo.get("unspecified"):
            parts.append(f"unspecified={stereo['unspecified']}")
        stereo_line = f"- Stereocenters: {stereo_total} ({', '.join(parts)})" if parts else stereo_line
    return "\n".join(
        [
            "Physicochemical Properties:",
            f"- Molecular weight: {_format_float(values.get('MolWt'), 2, ' Da')}",
            f"- Heavy atoms: {values.get('HeavyAtomCount')}, Heteroatoms: {values.get('NumHeteroatoms')}",
            f"- logP (Wildman-Crippen): {_format_float(values.get('MolLogP'))}",
            f"- TPSA: {_format_float(values.get('TPSA'), 2, ' A^2')}",
            f"- H-bond donors: {values.get('NumHDonors')}, H-bond acceptors: {values.get('NumHAcceptors')}",
            f"- Rotatable bonds: {values.get('NumRotatableBonds')}",
            f"- Fraction sp3 carbons (Fsp3): {_format_float(values.get('FractionCSP3'))}",
            f"- Molar refractivity: {_format_float(values.get('MolMR'))}",
            f"- Neutral fraction at pH 7.4: {_format_float(values.get('neutral_fraction_7_4'), 4)}",
            f"- Labute surface area: {_format_float(values.get('LabuteASA'), 4)}",
            f"- Nitrogen and oxygen atom count: {values.get('NOCount')}",
            (
                "- Carbocycle counts: "
                f"aliphatic={values.get('NumAliphaticCarbocycles')}, "
                f"aromatic={values.get('NumAromaticCarbocycles')}, "
                f"saturated={values.get('NumSaturatedCarbocycles')}"
            ),
            "",
            "Complexity Metrics:",
            f"- Bertz complexity: {_format_float(values.get('BertzCT'))}",
            f"- Balaban J: {_format_float(values.get('BalabanJ'))}",
            f"- Hall-Kier alpha: {_format_float(values.get('HallKierAlpha'))}",
            (
                "- Kappa shape indices: "
                f"K1={_format_float(values.get('Kappa1'))}, "
                f"K2={_format_float(values.get('Kappa2'))}, "
                f"K3={_format_float(values.get('Kappa3'))}"
            ),
            stereo_line,
            "",
            "Electronic Properties:",
            f"- Charge polarization: {_format_float(values.get('charge_polarization'))}",
            f"- Max absolute partial charge: {_format_float(values.get('MaxAbsPartialCharge'))}",
            f"- Min absolute partial charge: {_format_float(values.get('MinAbsPartialCharge'))}",
        ]
    )


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


def _compute_structure_text(canonical: str) -> tuple[dict[str, Any], str]:
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    from therapeutic_tools.utils.endpoints import functional_group_names

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise ValueError(f"invalid SMILES: {canonical!r}")
    functional_groups = functional_group_names(canonical) or []
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    values = {
        "functional_groups": functional_groups,
        "RingCount": int(Lipinski.RingCount(mol)),
        "NumAromaticRings": int(Lipinski.NumAromaticRings(mol)),
        "NumAliphaticRings": int(Lipinski.NumAliphaticRings(mol)),
        "NumSaturatedRings": int(Lipinski.NumSaturatedRings(mol)),
        "NumHeterocycles": int(Lipinski.NumHeterocycles(mol)),
        "largest_ring_size": max((len(ring) for ring in atom_rings), default=0),
    }
    text = "\n".join(
        [
            "Functional Groups:",
            f"- {', '.join(functional_groups) if functional_groups else 'none detected'}",
            "",
            "Ring Systems:",
            f"- Total rings: {values['RingCount']}",
            f"- Aromatic rings: {values['NumAromaticRings']}",
            f"- Aliphatic rings: {values['NumAliphaticRings']}",
            f"- Saturated rings: {values['NumSaturatedRings']}",
            f"- Heterocycles: {values['NumHeterocycles']}",
            f"- Largest ring size: {values['largest_ring_size']}",
        ]
    )
    return values, text


def _compute_alert_text(canonical: str) -> tuple[dict[str, Any], str]:
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
    if not merged:
        return {"categories": [], "category_count": 0}, "Structural Alerts:\nNo structural alerts found."
    lines = [f"Structural Alerts ({len(merged)} categories):"]
    for idx, (category, data) in enumerate(merged.items(), 1):
        alerts = data["alerts"]
        shown = alerts[:8]
        trail = f", ... and {len(alerts) - 8} more" if len(alerts) > 8 else ""
        lines.extend(["", f"{idx}. {category}", str(data["note"]), f"Matched alerts ({len(alerts)}): {', '.join(shown)}{trail}"])
    return {"categories": list(merged), "category_count": len(merged)}, "\n".join(lines)


def compute_openrlhf_records(smiles: str) -> list[dict[str, Any]]:
    info = canonicalize_smiles(smiles)
    if info["canonicalization_status"] != "ok":
        return [
            endpoint_record(
                smiles_info=info,
                endpoint_name=endpoint,
                runtime_name="openrlhf",
                status="error",
                error=info["canonicalization_error"],
            )
            for endpoint in OPENRLHF_ENDPOINTS
        ]

    canonical = str(info["canonical_smiles"])
    outputs = {
        "v17_molecular_profile_text": lambda: (
            (values := primitive_compute._compute_rdkit_profile(canonical)),
            renderers.render_molecule_profile(values),
        ),
        "v17_pka_ionization_logd_text": lambda: (
            (values := primitive_compute._compute_pka_ionization_logd(canonical)),
            values["text"],
        ),
        "v17_structure_and_topology_text": lambda: (
            (values := primitive_compute._compute_structure_values(canonical)),
            renderers.render_structure_and_topology(values),
        ),
        "v17_alert_screening_text": lambda: (
            (values := primitive_compute._compute_alert_values(canonical)),
            renderers.render_alert_screening(values),
        ),
    }
    records: list[dict[str, Any]] = []
    for endpoint, compute in outputs.items():
        try:
            require_runtime_for_endpoint(endpoint)
            value, text = compute()
            text = str(text)
            reason = bad_text_reason(text)
            if reason is not None:
                records.append(
                    endpoint_record(
                        smiles_info=info,
                        endpoint_name=endpoint,
                        runtime_name="openrlhf",
                        status="error",
                        text=text,
                        value=value,
                        error=reason,
                    )
                )
            else:
                records.append(
                    endpoint_record(
                        smiles_info=info,
                        endpoint_name=endpoint,
                        runtime_name="openrlhf",
                        status="ok",
                        value=value,
                        text=text,
                    )
                )
        except Exception as exc:
            records.append(
                endpoint_record(
                    smiles_info=info,
                    endpoint_name=endpoint,
                    runtime_name="openrlhf",
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return records


def compute_minimol_records(smiles_list: list[str], batch_size: int = 64) -> Iterator[dict[str, Any]]:
    """Compatibility wrapper for callers that still import this helper."""
    yield from primitive_compute.compute_minimol_records(smiles_list, batch_size=batch_size)


def compute_shard(runtime_name: str, input_smiles: Path, output_jsonl: Path, batch_size: int = 64) -> dict[str, Any]:
    smiles = [str(row["smiles"]) for row in read_jsonl(input_smiles)]
    if runtime_name == "openrlhf":
        rows = (record for smi in smiles for record in compute_openrlhf_records(smi))
    elif runtime_name == "minimol":
        rows = primitive_compute.compute_minimol_records(smiles, batch_size=batch_size)
    else:
        raise ValueError(f"unsupported runtime {runtime_name!r}")
    count = write_records(output_jsonl, rows)
    return {"runtime": runtime_name, "input_smiles": len(smiles), "records": count, "output_jsonl": str(output_jsonl)}


def init_feature_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            smiles TEXT PRIMARY KEY,
            feature_text TEXT,
            status TEXT NOT NULL,
            error TEXT,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_status ON features(status)")
    conn.commit()
    return conn


def load_cached_smiles(conn: sqlite3.Connection, smiles: Iterable[str]) -> set[str]:
    values = list(smiles)
    cached: set[str] = set()
    for start in range(0, len(values), 900):
        chunk = values[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        cursor = conn.execute(f"SELECT smiles FROM features WHERE status = 'ok' AND smiles IN ({placeholders})", chunk)
        cached.update(str(row[0]) for row in cursor)
    return cached


def records_by_smiles(record_paths: Iterable[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in record_paths:
        if not path.exists():
            continue
        for row in read_records(path):
            endpoint = row.get("endpoint_name")
            if not endpoint:
                continue
            for key in (row.get("raw_smiles"), row.get("canonical_smiles")):
                if key:
                    grouped[str(key)][str(endpoint)] = row
    return grouped


def assemble_feature_text(endpoint_rows: dict[str, dict[str, Any]], feature_names: tuple[str, ...]) -> tuple[str, str | None]:
    required = {
        "molecular_profile": ("v17_molecular_profile_text",),
        "ionization_and_solubility": ("v17_pka_ionization_logd_text", "minimol_solubility_log_s"),
        "structure_and_topology": ("v17_structure_and_topology_text",),
        "alert_screening": ("v17_alert_screening_text",),
    }
    sections: list[str] = []
    errors: list[str] = []
    for feature_name in feature_names:
        for endpoint in required[feature_name]:
            row = endpoint_rows.get(endpoint)
            if row is None:
                errors.append(f"{endpoint}:missing")
                continue
            if row.get("status") != "ok":
                errors.append(f"{endpoint}:{row.get('error') or 'error'}")
                continue
            text = str(row.get("text") or "")
            reason = bad_text_reason(text)
            if reason is not None:
                errors.append(f"{endpoint}:{reason}")
                continue
            sections.append(text)
    if errors:
        return "", "; ".join(errors)
    return "\n\n".join(section for section in sections if section), None


def split_smiles(smiles: list[str], chunks: int) -> list[list[str]]:
    chunks = max(1, chunks)
    return [smiles[i::chunks] for i in range(chunks) if smiles[i::chunks]]


def run_runtime_workers(
    *,
    runtime_name: str,
    smiles: list[str],
    work_dir: Path,
    workers: int,
    batch_size: int = 64,
) -> list[Path]:
    runtime = get_runtime(runtime_name)
    worker_chunks = split_smiles(smiles, workers)
    outputs: list[Path] = []
    procs: list[tuple[subprocess.Popen[str], Path]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONNOUSERSITE"] = "1"
    env["THERAPEUTIC_TOOLS_DISABLE_STRUCTURED_FEATURE_CACHE"] = "1"
    env["THERAPEUTIC_TOOLS_CACHE_BACKEND"] = DUCKDB_ONLY_BACKEND
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-therapeutic-tools")
    for idx, chunk in enumerate(worker_chunks):
        input_path = work_dir / f"{runtime_name}_{idx:04d}.smiles.jsonl"
        output_path = work_dir / f"{runtime_name}_{idx:04d}.records.parquet"
        write_jsonl(input_path, ({"smiles": smi} for smi in chunk))
        cmd = [
            str(runtime.python),
            "-m",
            "therapeutic_tools.runtime_cache",
            "compute-shard",
            "--runtime",
            runtime_name,
            "--input-smiles",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--batch-size",
            str(batch_size),
        ]
        procs.append((subprocess.Popen(cmd, env=env, text=True), output_path))
        outputs.append(output_path)
    for proc, output_path in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{runtime_name} worker failed with exit code {rc}: {output_path}")
    return outputs


def load_cached_duckdb_smiles(duckdb_path: Path | None, smiles: Iterable[str], feature_names: tuple[str, ...]) -> set[str]:
    if duckdb_path is None or not duckdb_path.exists():
        return set()
    import duckdb

    bundle_aliases = {"molecular_profile": "molecule_profile"}
    feature_key = ",".join(bundle_aliases.get(name, name) for name in feature_names)
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        has_table = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'full_call_strings'"
        ).fetchone()[0]
        if not has_table:
            return set()
        values = list(dict.fromkeys(str(smi) for smi in smiles if str(smi).strip()))
        cached: set[str] = set()
        for start in range(0, len(values), 5000):
            chunk = values[start : start + 5000]
            rows = conn.execute(
                """
                SELECT raw_smiles
                FROM full_call_strings
                WHERE status = 'ok'
                  AND feature_set_key = ?
                  AND raw_smiles IN (SELECT * FROM UNNEST(?))
                """,
                [feature_key, chunk],
            ).fetchall()
            cached.update(str(row[0]) for row in rows)
        return cached
    finally:
        conn.close()


def build_full_cache(
    *,
    smiles: Iterable[str],
    work_dir: Path,
    feature_names: tuple[str, ...] = FEATURE_GROUPS,
    workers: int = 8,
    minimol_workers: int | None = None,
    batch_size: int = 64,
    duckdb_path: Path,
) -> dict[str, Any]:
    smiles_list = sorted(dict.fromkeys(str(smi) for smi in smiles if str(smi).strip()))
    cached = load_cached_duckdb_smiles(duckdb_path, smiles_list, feature_names)
    missing = [smi for smi in smiles_list if smi not in cached]
    stats: dict[str, Any] = {
        "duckdb_path": str(duckdb_path),
        "total_smiles": len(smiles_list),
        "cached_ok_before": len(cached),
        "missing_before": len(missing),
        "feature_names": list(feature_names),
        "runtime_aware": True,
        "cache_backend": DUCKDB_ONLY_BACKEND,
        "runtime_records": {},
    }
    if not missing:
        stats.update({"computed_ok": 0, "computed_error": 0})
        return stats
    work_dir.mkdir(parents=True, exist_ok=True)
    openrlhf_workers = max(1, int(workers))
    if minimol_workers is None:
        minimol_workers = max(1, min(4, max(1, int(workers)) // 4))
    record_paths: list[Path] = []
    openrlhf_paths = run_runtime_workers(
        runtime_name="openrlhf",
        smiles=missing,
        work_dir=work_dir,
        workers=openrlhf_workers,
        batch_size=batch_size,
    )
    record_paths.extend(openrlhf_paths)
    stats["runtime_records"]["openrlhf"] = [str(path) for path in openrlhf_paths]
    if "ionization_and_solubility" in feature_names or "concise_profile" in feature_names:
        minimol_paths = run_runtime_workers(
            runtime_name="minimol",
            smiles=missing,
            work_dir=work_dir,
            workers=max(1, int(minimol_workers)),
            batch_size=batch_size,
        )
        record_paths.extend(minimol_paths)
        stats["runtime_records"]["minimol"] = [str(path) for path in minimol_paths]
    assembly = run_duckdb_publish(
        duckdb_path=duckdb_path,
        smiles=missing,
        record_paths=record_paths,
        feature_names=feature_names,
    )
    stats.update(assembly)
    stats["computed_ok"] = int(assembly.get("full_call_ok", 0))
    stats["computed_error"] = int(assembly.get("full_call_error", 0))
    return stats


def audit_legacy(paths: list[Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {"sources": {}}
    for path in paths:
        source = Counter()
        if not path.exists():
            source["missing_source"] += 1
        elif path.suffix == ".jsonl":
            for row in read_jsonl(path):
                text = row.get("result") or row.get("feature_text") or row.get("text")
                reason = bad_text_reason(str(text) if text is not None else None)
                source["rejected" if reason else "accepted_candidate"] += 1
                if reason:
                    source[reason] += 1
        elif path.suffix in {".sqlite", ".db"}:
            conn = sqlite3.connect(str(path))
            try:
                tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                source["tables"] = len(tables)
                for table in tables:
                    cols = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
                    text_col = next((col for col in ("feature_text", "result", "text", "value_json", "error") if col in cols), None)
                    if text_col is None:
                        continue
                    for row in conn.execute(f"SELECT {text_col} FROM {table} LIMIT 100000"):
                        reason = bad_text_reason(str(row[0]) if row[0] is not None else None)
                        source["rejected" if reason else "accepted_candidate"] += 1
                        if reason:
                            source[reason] += 1
            finally:
                conn.close()
        else:
            source["unsupported_source_type"] += 1
        stats["sources"][str(path)] = dict(source)
    return stats


def import_legacy_feature_sqlite(source_path: Path, target_path: Path, *, batch_size: int = 5000) -> dict[str, Any]:
    """Import compatible legacy full-call feature rows into a compatibility SQLite cache.

    This is intentionally conservative and intended for compatibility/debug
    cache views. Production MiniMol solubility caches should recompute full-call
    text from primitive rows instead of relying on legacy full-call strings.
    """
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_conn = sqlite3.connect(str(source_path))
    target_conn = init_feature_db(target_path)
    stats = Counter({"source_cache_exists": 1})
    try:
        source_cols = {
            str(row[1])
            for row in source_conn.execute("PRAGMA table_info(features)")
        }
        required = {"smiles", "feature_text", "status", "error", "updated_at_utc"}
        if not required.issubset(source_cols):
            raise ValueError(f"{source_path} does not look like a legacy features SQLite cache")
        cursor = source_conn.execute(
            """
            SELECT smiles, feature_text, status, error, updated_at_utc
            FROM features
            WHERE status = 'ok'
            ORDER BY smiles
            """
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            accepted = []
            for smiles, feature_text, status, error, updated_at in rows:
                reason = bad_text_reason(str(feature_text) if feature_text is not None else None)
                if reason is not None:
                    stats["rejected"] += 1
                    stats[reason] += 1
                    continue
                accepted.append((smiles, feature_text, status, error, updated_at))
                stats["accepted"] += 1
            if accepted:
                before = target_conn.total_changes
                target_conn.executemany(
                    """
                    INSERT OR IGNORE INTO features(smiles, feature_text, status, error, updated_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    accepted,
                )
                stats["inserted"] += target_conn.total_changes - before
                target_conn.commit()
        return {
            "source_cache": str(source_path),
            "target_cache": str(target_path),
            "compatibility_only": True,
            **dict(stats),
        }
    finally:
        source_conn.close()
        target_conn.close()


def import_legacy_to_duckdb(source_path: Path, duckdb_path: Path) -> dict[str, Any]:
    """Audit a legacy source into DuckDB without exposing it to production reads."""
    import duckdb

    audit = audit_legacy([source_path])
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(duckdb_path))
    now = utc_now()
    try:
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
        source_stats = audit["sources"].get(str(source_path), {})
        source_kind = source_path.suffix.lstrip(".") or "unknown"
        for reason, count in source_stats.items():
            if not isinstance(count, int):
                continue
            if reason.startswith("bad_text_marker") or reason in {"empty_text", "missing_source", "unsupported_source_type"}:
                decision = "reject"
            elif reason == "accepted_candidate":
                decision = "audit_only"
            else:
                decision = "metadata"
            conn.execute(
                """
                INSERT INTO legacy_audit(source_path, source_kind, decision, reason, count, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [str(source_path), source_kind, decision, reason, int(count), now],
            )
        return {"duckdb_path": str(duckdb_path), "source_path": str(source_path), "audit": audit}
    finally:
        conn.close()


def _feature_requirements(feature_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "molecular_profile": ("v17_molecular_profile_text",),
        "ionization_and_solubility": ("v17_pka_ionization_logd_text", "minimol_solubility_log_s"),
        "structure_and_topology": ("v17_structure_and_topology_text",),
        "alert_screening": ("v17_alert_screening_text",),
    } | {name: () for name in FEATURE_GROUPS if name not in feature_names}


def endpoint_record_to_primitive_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Explode runtime endpoint records into atomic primitive rows."""
    from therapeutic_tools import duckdb_store as store
    from therapeutic_tools import primitive_registry as registry

    info = {
        "raw_smiles": row.get("raw_smiles"),
        "canonical_smiles": row.get("canonical_smiles"),
        "raw_is_canonical": row.get("raw_is_canonical"),
        "canonicalization_status": row.get("canonicalization_status") or "ok",
        "canonicalization_error": row.get("canonicalization_error"),
    }
    endpoint = str(row.get("endpoint_name") or "")
    runtime_name = str(row.get("runtime_name") or "")
    value = value_as_dict(row.get("value_json"))
    status = str(row.get("status") or "error")
    error = row.get("error")

    def make(name: str, primitive_value: Any, method: str) -> dict[str, Any]:
        return {
            **info,
            "primitive_name": name,
            "params_json": "{}",
            "value_type": store.value_type(primitive_value),
            "value_json": store.encode_value(primitive_value),
            "status": status,
            "error": error,
            "runtime_name": runtime_name,
            "method": method,
            "provenance_json": store.encode_value(
                {
                    "endpoint_name": endpoint,
                    "endpoint_version": row.get("endpoint_version"),
                    "runtime_name": runtime_name,
                    "method": method,
                }
            ),
            "updated_at_utc": row.get("updated_at_utc") or utc_now(),
        }

    if endpoint == "v17_molecular_profile_text":
        return [make(name, value.get(name), "RDKit descriptors") for name in registry.MOLECULE_PROFILE_PRIMITIVES]
    if endpoint == "v17_pka_ionization_logd_text":
        return [
            make("pka_summary", value.get("pka"), "MolGpKa/RDKit"),
            make("ionization_text", value.get("ionization_text"), "MolGpKa/RDKit"),
            make("logD_74", value.get("logD_74"), "MolGpKa/RDKit"),
            make("pka_ionization_logd_text", value.get("text") or row.get("text"), "MolGpKa/RDKit"),
        ]
    if endpoint == "minimol_solubility_log_s":
        return [make("minimol_solubility_log_s", value.get("log_s"), "MiniMol + Ridge (AqSolDB)")]
    if endpoint == "v17_structure_and_topology_text":
        remap = {
            "functional_groups": "functional_groups",
            "ring_count": "ring_count",
            "aromatic_rings": "aromatic_rings",
            "aliphatic_rings": "aliphatic_rings",
            "saturated_rings": "saturated_rings",
            "heterocycles": "heterocycles",
            "largest_ring_size": "largest_ring_size",
        }
        return [make(name, value.get(key), "RDKit SMARTS/ring topology") for name, key in remap.items()]
    if endpoint == "v17_alert_screening_text":
        return [
            make("alert_categories", value.get("categories") or [], "RDKit structural alerts"),
            make("alert_category_count", value.get("category_count") or 0, "RDKit structural alerts"),
        ]
    return []


def publish_duckdb(
    record_paths: list[Path],
    duckdb_path: Path,
    *,
    smiles: Iterable[str] | None = None,
    feature_names: tuple[str, ...] = FEATURE_GROUPS,
) -> dict[str, Any]:
    import duckdb

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(duckdb_path))
    try:
        def write_atomic_rows_bulk(rows: list[dict[str, Any]]) -> None:
            if not rows:
                return
            import pandas as pd

            columns = [
                "raw_smiles",
                "canonical_smiles",
                "raw_is_canonical",
                "canonicalization_status",
                "canonicalization_error",
                "primitive_name",
                "endpoint_version",
                "python_executable",
                "params_json",
                "value_type",
                "runtime_name",
                "status",
                "value_json",
                "error",
                "method",
                "provenance_json",
                "updated_at_utc",
            ]
            normalized = []
            for row in rows:
                item = {column: row.get(column) for column in columns}
                item["params_json"] = item.get("params_json") or "{}"
                item["status"] = item.get("status") or "ok"
                item["canonicalization_status"] = item.get("canonicalization_status") or "ok"
                item["updated_at_utc"] = item.get("updated_at_utc") or utc_now()
                normalized.append(item)
            df = pd.DataFrame(normalized, columns=columns)
            conn.register("atomic_rows_df", df)
            try:
                conn.execute("CREATE TEMP TABLE atomic_rows AS SELECT * FROM atomic_rows_df")
                conn.execute(
                    """
                    DELETE FROM primitive_values
                    USING (
                        SELECT DISTINCT raw_smiles, primitive_name, COALESCE(params_json, '{}') AS params_json
                        FROM atomic_rows
                    ) affected
                    WHERE primitive_values.raw_smiles = affected.raw_smiles
                      AND primitive_values.primitive_name = affected.primitive_name
                      AND COALESCE(primitive_values.params_json, '{}') = affected.params_json
                    """
                )
                conn.execute(
                    """
                    INSERT INTO primitive_values(
                        raw_smiles, canonical_smiles, raw_is_canonical,
                        canonicalization_status, canonicalization_error,
                        primitive_name, endpoint_version, python_executable,
                        params_json, value_type, runtime_name, status, value_json,
                        error, method, provenance_json, updated_at_utc
                    )
                    SELECT
                        raw_smiles,
                        canonical_smiles,
                        raw_is_canonical,
                        canonicalization_status,
                        canonicalization_error,
                        primitive_name,
                        endpoint_version,
                        python_executable,
                        params_json,
                        value_type,
                        runtime_name,
                        status,
                        value_json::JSON,
                        error,
                        method,
                        provenance_json::JSON,
                        updated_at_utc
                    FROM atomic_rows
                    """
                )
                conn.execute("DROP TABLE atomic_rows")
            finally:
                conn.unregister("atomic_rows_df")

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
                runtime_name VARCHAR,
                status VARCHAR,
                value_json JSON,
                error VARCHAR,
                method VARCHAR,
                provenance_json JSON,
                updated_at_utc VARCHAR
            )
            """
        )
        existing_cols = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'primitive_values'
                """
            ).fetchall()
        }
        for column, sql_type in (
            ("params_json", "VARCHAR"),
            ("value_type", "VARCHAR"),
            ("method", "VARCHAR"),
            ("provenance_json", "JSON"),
        ):
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE primitive_values ADD COLUMN {column} {sql_type}")
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
        inserted = 0
        bundle_aliases = {"molecular_profile": "molecule_profile"}
        normalized_feature_names = tuple(bundle_aliases.get(name, name) for name in feature_names)
        from therapeutic_tools.primitive_registry import BUNDLE_PRIMITIVES

        required_primitive_names = {
            primitive
            for feature_name in normalized_feature_names
            for primitive in BUNDLE_PRIMITIVES[feature_name]
        }
        for path in record_paths:
            records = list(read_records(path))
            inserted += len(records)
            atomic_rows = [
                primitive
                for record in records
                for primitive in endpoint_record_to_primitive_rows(record)
                if primitive.get("primitive_name") in required_primitive_names
            ]
            write_atomic_rows_bulk(atomic_rows)

        concise_only = tuple(normalized_feature_names) == ("concise_profile",)
        grouped = {} if concise_only and smiles is not None else records_by_smiles(record_paths)
        source_smiles = list(dict.fromkeys(str(smi) for smi in (smiles or grouped.keys()) if str(smi).strip()))
        feature_key = ",".join(normalized_feature_names)
        primitive_rows = int(conn.execute("SELECT COUNT(*) FROM primitive_values").fetchone()[0])
        stats = Counter({"inserted_records": inserted, "primitive_rows": primitive_rows})
        now = utc_now()

        for smi in source_smiles:
            info = canonicalize_smiles(smi)
            conn.execute(
                """
                INSERT OR REPLACE INTO molecules(
                    raw_smiles, canonical_smiles, raw_is_canonical,
                    canonicalization_status, canonicalization_error, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    info.get("raw_smiles"),
                    info.get("canonical_smiles"),
                    info.get("raw_is_canonical"),
                    info.get("canonicalization_status"),
                    info.get("canonicalization_error"),
                    now,
                ],
            )
            endpoint_rows = grouped.get(smi)
            if endpoint_rows is None:
                canonical = info.get("canonical_smiles")
                endpoint_rows = grouped.get(str(canonical), {}) if canonical else {}

            full_sections: list[str] = []
            full_errors: list[str] = []
            provenance = {"feature_names": list(feature_names), "required_endpoints": {}, "endpoints": {}}
            for feature_name in feature_names:
                sections: list[str] = []
                errors: list[str] = []
                feature_provenance: dict[str, Any] = {"required_endpoints": [], "endpoints": {}}
                if feature_name == "concise_profile":
                    from therapeutic_tools import duckdb_store as store
                    from therapeutic_tools.primitive_registry import BUNDLE_PRIMITIVES

                    primitive_names = BUNDLE_PRIMITIVES["concise_profile"]
                    values, missing_primitives = store.lookup_primitives(conn, info, primitive_names)
                    feature_provenance = {"required_primitives": list(primitive_names)}
                    provenance.setdefault("required_primitives", {})[feature_name] = list(primitive_names)
                    if missing_primitives:
                        errors.extend(f"{name}:missing" for name in missing_primitives)
                    semantic_status = "error" if errors else "ok"
                    semantic_text = renderers.render_bundle("concise_profile", values) if not errors else ""
                    semantic_error = "; ".join(errors) if errors else None
                else:
                    required = {
                        "molecular_profile": ("v17_molecular_profile_text",),
                        "ionization_and_solubility": ("v17_pka_ionization_logd_text", "minimol_solubility_log_s"),
                        "structure_and_topology": ("v17_structure_and_topology_text",),
                        "alert_screening": ("v17_alert_screening_text",),
                    }[feature_name]
                    provenance["required_endpoints"][feature_name] = list(required)
                    feature_provenance["required_endpoints"] = list(required)
                    for endpoint in required:
                        row = endpoint_rows.get(endpoint)
                        if row is None:
                            errors.append(f"{endpoint}:missing")
                            continue
                        if row.get("status") != "ok":
                            errors.append(f"{endpoint}:{row.get('error') or 'error'}")
                            continue
                        text = str(row.get("text") or "")
                        reason = bad_text_reason(text)
                        if reason is not None:
                            errors.append(f"{endpoint}:{reason}")
                            continue
                        endpoint_provenance = {
                            "runtime_name": row.get("runtime_name"),
                            "endpoint_version": row.get("endpoint_version"),
                        }
                        value = value_as_dict(row.get("value_json"))
                        if endpoint == "minimol_solubility_log_s":
                            method = value.get("method")
                            endpoint_provenance["method"] = method
                            if method != "MiniMol + Ridge (AqSolDB)":
                                errors.append(f"{endpoint}:missing_minimol_provenance")
                                continue
                        provenance["endpoints"][endpoint] = endpoint_provenance
                        feature_provenance["endpoints"][endpoint] = endpoint_provenance
                        sections.append(text)
                    semantic_status = "error" if errors else "ok"
                    semantic_text = "\n\n".join(sections) if not errors else ""
                    semantic_error = "; ".join(errors) if errors else None
                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_strings(
                        raw_smiles, canonical_smiles, feature_name, status,
                        text, error, provenance_json, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?)
                    """,
                    [
                        smi,
                        info.get("canonical_smiles"),
                        feature_name,
                        semantic_status,
                        semantic_text,
                        semantic_error,
                        json.dumps(feature_provenance, sort_keys=True),
                        now,
                    ],
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO bundle_strings(
                        raw_smiles, canonical_smiles, bundle_name, status,
                        text, error, provenance_json, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?)
                    """,
                    [
                        smi,
                        info.get("canonical_smiles"),
                        bundle_aliases.get(feature_name, feature_name),
                        semantic_status,
                        semantic_text,
                        semantic_error,
                        json.dumps(feature_provenance, sort_keys=True),
                        now,
                    ],
                )
                stats[f"semantic_{semantic_status}"] += 1
                if errors:
                    full_errors.extend(errors)
                else:
                    full_sections.append(semantic_text)

            full_status = "error" if full_errors else "ok"
            full_text = "\n\n".join(section for section in full_sections if section) if not full_errors else ""
            conn.execute(
                """
                INSERT OR REPLACE INTO full_call_strings(
                    raw_smiles, canonical_smiles, feature_set_key, status,
                    feature_text, error, provenance_json, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?)
                """,
                [
                    smi,
                    info.get("canonical_smiles"),
                    feature_key,
                    full_status,
                    full_text,
                    "; ".join(full_errors) if full_errors else None,
                    json.dumps(provenance, sort_keys=True),
                    now,
                ],
            )
            stats[f"full_call_{full_status}"] += 1

        run_id = f"duckdb-only-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
        conn.execute(
            """
            INSERT INTO build_runs(
                run_id, cache_backend, feature_set_key, source_smiles_count,
                record_path_count, stats_json, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?::JSON, ?)
            """,
            [
                run_id,
                DUCKDB_ONLY_BACKEND,
                feature_key,
                len(source_smiles),
                len(record_paths),
                json.dumps(dict(stats), sort_keys=True),
                now,
            ],
        )
        return {"duckdb_path": str(duckdb_path), "feature_set_key": feature_key, "run_id": run_id, **dict(stats)}
    finally:
        conn.close()


def run_duckdb_publish(
    *,
    duckdb_path: Path,
    smiles: Iterable[str],
    record_paths: list[Path],
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    runtime = get_runtime("minimol")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONNOUSERSITE"] = "1"
    env["THERAPEUTIC_TOOLS_CACHE_BACKEND"] = DUCKDB_ONLY_BACKEND
    with tempfile.NamedTemporaryFile("wt", suffix=".smiles.jsonl", delete=False) as handle:
        smiles_path = Path(handle.name)
        for smi in smiles:
            handle.write(json.dumps({"smiles": str(smi)}, separators=(",", ":")) + "\n")
    common = [str(runtime.python), "-m", "therapeutic_tools.runtime_cache"]
    try:
        cmd = common + [
            "publish-duckdb",
            "--duckdb-path",
            str(duckdb_path),
            "--input-smiles",
            str(smiles_path),
            "--feature-names",
            *feature_names,
            "--",
            *[str(path) for path in record_paths],
        ]
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "DuckDB publish failed with exit code "
                f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        result = json.loads(proc.stdout)
        return {"duckdb": result, **{k: v for k, v in result.items() if k.startswith(("full_call_", "semantic_", "inserted_"))}}
    finally:
        try:
            smiles_path.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser("compute-shard")
    p_compute.add_argument("--runtime", choices=("openrlhf", "minimol"), required=True)
    p_compute.add_argument("--input-smiles", type=Path, required=True)
    p_compute.add_argument("--output-jsonl", type=Path, required=True)
    p_compute.add_argument("--batch-size", type=int, default=64)

    p_build = sub.add_parser("build-full-cache")
    p_build.add_argument("--input-smiles", type=Path, required=True)
    p_build.add_argument("--work-dir", type=Path, required=True)
    p_build.add_argument("--feature-names", nargs="+", choices=FEATURE_GROUPS, default=list(FEATURE_GROUPS))
    p_build.add_argument("--workers", type=int, default=8)
    p_build.add_argument("--minimol-workers", type=int, default=None)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--duckdb-path", type=Path, required=True)

    p_audit = sub.add_parser("audit-legacy")
    p_audit.add_argument("paths", type=Path, nargs="+")

    p_import = sub.add_parser("import-legacy")
    p_import.add_argument("--source-sqlite", type=Path, required=True)
    p_import.add_argument("--target-duckdb", type=Path, required=True)

    p_duck = sub.add_parser("publish-duckdb")
    p_duck.add_argument("--duckdb-path", type=Path, required=True)
    p_duck.add_argument("--input-smiles", type=Path, default=None)
    p_duck.add_argument("--feature-names", nargs="+", choices=FEATURE_GROUPS, default=list(FEATURE_GROUPS))
    p_duck.add_argument("record_paths", type=Path, nargs="+")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compute-shard":
        result = compute_shard(args.runtime, args.input_smiles, args.output_jsonl, batch_size=args.batch_size)
    elif args.command == "build-full-cache":
        smiles = [str(row["smiles"]) for row in read_jsonl(args.input_smiles)]
        result = build_full_cache(
            smiles=smiles,
            work_dir=args.work_dir,
            feature_names=tuple(args.feature_names),
            workers=args.workers,
            minimol_workers=args.minimol_workers,
            batch_size=args.batch_size,
            duckdb_path=args.duckdb_path,
        )
    elif args.command == "audit-legacy":
        result = audit_legacy(args.paths)
    elif args.command == "import-legacy":
        result = import_legacy_to_duckdb(args.source_sqlite, args.target_duckdb)
    elif args.command == "publish-duckdb":
        smiles = [str(row["smiles"]) for row in read_jsonl(args.input_smiles)] if args.input_smiles else None
        result = publish_duckdb(
            args.record_paths,
            args.duckdb_path,
            smiles=smiles,
            feature_names=tuple(args.feature_names),
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
