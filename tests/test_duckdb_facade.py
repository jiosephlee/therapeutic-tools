from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from therapeutic_tools import api, duckdb_store, primitive_compute
from therapeutic_tools.primitive_registry import BUNDLE_PRIMITIVES


pytestmark = pytest.mark.skipif(importlib.util.find_spec("duckdb") is None, reason="duckdb is not installed")


def _row(smiles: str, name: str, value):
    info = duckdb_store.canonicalize_smiles(smiles)
    return primitive_compute.primitive_row(info, name, value, "test", "test method")


def test_primitive_miss_computes_and_writes_atomic_row(tmp_path, monkeypatch):
    db_path = tmp_path / "features.duckdb"
    monkeypatch.setenv("THERAPEUTIC_TOOLS_DUCKDB_CACHE", str(db_path))

    def fake_compute(smiles, primitive_names):
        assert list(primitive_names) == ["MolWt"]
        return [_row(smiles, "MolWt", 46.07)]

    monkeypatch.setattr(primitive_compute, "compute_missing_primitives", fake_compute)

    assert api.get_primitive("CCO", "MolWt") == 46.07

    conn = duckdb_store.connect(db_path, read_only=True)
    try:
        rows = conn.execute(
            "SELECT primitive_name, status, value_json FROM primitive_values"
        ).fetchall()
    finally:
        conn.close()
    assert [(row[0], row[1]) for row in rows] == [("MolWt", "ok")]


def test_raw_lookup_precedes_canonical_fallback(tmp_path, monkeypatch):
    db_path = tmp_path / "features.duckdb"
    monkeypatch.setenv("THERAPEUTIC_TOOLS_DUCKDB_CACHE", str(db_path))
    conn = duckdb_store.connect(db_path)
    try:
        duckdb_store.write_primitive_rows(
            conn,
            [
                _row("OCC", "MolWt", 1.0),
                _row("CCO", "MolWt", 2.0),
            ],
        )
    finally:
        conn.close()

    assert api.get_primitive("OCC", "MolWt") == 1.0


def test_bundle_renders_from_primitive_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "features.duckdb"
    monkeypatch.setenv("THERAPEUTIC_TOOLS_DUCKDB_CACHE", str(db_path))
    conn = duckdb_store.connect(db_path)
    try:
        duckdb_store.write_primitive_rows(
            conn,
            [
                _row("CCO", "functional_groups", ["alcohol"]),
                _row("CCO", "ring_count", 0),
                _row("CCO", "aromatic_rings", 0),
                _row("CCO", "aliphatic_rings", 0),
                _row("CCO", "saturated_rings", 0),
                _row("CCO", "heterocycles", 0),
                _row("CCO", "largest_ring_size", 0),
            ],
        )
    finally:
        conn.close()

    text = api.get_bundle("CCO", "structure_and_topology")
    assert "Functional Groups:" in text
    assert "alcohol" in text
    assert "Ring Systems:" in text

    conn = duckdb_store.connect(db_path, read_only=True)
    try:
        stored = conn.execute(
            "SELECT bundle_name, status FROM bundle_strings"
        ).fetchall()
    finally:
        conn.close()
    assert stored == [("structure_and_topology", "ok")]


def test_all_features_stores_and_reads_full_call_string(tmp_path, monkeypatch):
    db_path = tmp_path / "features.duckdb"
    monkeypatch.setenv("THERAPEUTIC_TOOLS_DUCKDB_CACHE", str(db_path))
    calls = []

    def fake_compute(smiles, primitive_names):
        calls.append(tuple(primitive_names))
        rows = []
        for name in primitive_names:
            if name == "functional_groups":
                value = ["alcohol"]
            elif name == "alert_categories":
                value = []
            elif name == "stereocenter_breakdown":
                value = {"r": 0, "s": 0, "unspecified": 0}
            elif name == "pka_summary":
                value = {"acidic": [], "basic": []}
            elif name == "ionization_text":
                value = "Ionization at pH 7.4: neutral, charge 0"
            elif name == "pka_ionization_logd_text":
                value = "pKa: none\n\nIonization at pH 7.4: neutral, charge 0\n\nLogD at pH 7.4: 0.00"
            elif name == "minimol_solubility_log_s":
                value = -0.42
            else:
                value = 1
            rows.append(_row(smiles, name, value))
        return rows

    monkeypatch.setattr(primitive_compute, "compute_missing_primitives", fake_compute)

    text = api.get_all_features("CCO")
    assert "Physicochemical Properties:" in text
    assert "Solubility: logS = -0.42 log(mol/L)" in text

    conn = duckdb_store.connect(db_path, read_only=True)
    try:
        full_rows = conn.execute(
            "SELECT feature_set_key, status, feature_text FROM full_call_strings"
        ).fetchall()
    finally:
        conn.close()
    assert len(full_rows) == 1
    assert full_rows[0][0] == "molecule_profile,ionization_and_solubility,structure_and_topology,alert_screening"
    assert full_rows[0][1] == "ok"

    monkeypatch.setattr(
        primitive_compute,
        "compute_missing_primitives",
        lambda smiles, primitive_names: pytest.fail("full-call cache should be reused"),
    )
    assert api.get_all_features("CCO") == full_rows[0][2]
    assert calls


def test_minimol_method_provenance_is_not_visible_in_bundle_text(tmp_path, monkeypatch):
    db_path = tmp_path / "features.duckdb"
    monkeypatch.setenv("THERAPEUTIC_TOOLS_DUCKDB_CACHE", str(db_path))
    rows = [
        _row("CCO", "pka_summary", {"acidic": [], "basic": []}),
        _row("CCO", "ionization_text", "Ionization at pH 7.4: neutral, charge 0"),
        _row("CCO", "logD_74", 0.0),
        _row("CCO", "pka_ionization_logd_text", "pKa: none\n\nLogD at pH 7.4: 0.00"),
        _row("CCO", "minimol_solubility_log_s", -0.5),
    ]
    rows[-1]["method"] = "MiniMol + Ridge (AqSolDB)"
    rows[-1]["provenance_json"] = duckdb_store.encode_value(
        {"method": "MiniMol + Ridge (AqSolDB)", "runtime_name": "minimol"}
    )
    conn = duckdb_store.connect(db_path)
    try:
        duckdb_store.write_primitive_rows(conn, rows)
    finally:
        conn.close()

    text = api.get_bundle("CCO", "ionization_and_solubility")
    assert "Solubility: logS = -0.50 log(mol/L)" in text
    assert "MiniMol" not in text
    assert "AqSolDB" not in text

    conn = duckdb_store.connect(db_path, read_only=True)
    try:
        method = conn.execute(
            """
            SELECT method
            FROM primitive_values
            WHERE primitive_name = 'minimol_solubility_log_s'
            """
        ).fetchone()[0]
    finally:
        conn.close()
    assert method == "MiniMol + Ridge (AqSolDB)"


def test_public_v17_source_does_not_import_legacy_text_caches():
    source = Path(__file__).resolve().parents[1] / "tools" / "v17.py"
    text = source.read_text()
    assert "group_string_cache" not in text
    assert "metadata_cache" not in text
    assert "structured_feature_cache" not in text


def test_bundle_registry_has_primitive_dependencies():
    assert set(BUNDLE_PRIMITIVES) == {
        "molecule_profile",
        "ionization_and_solubility",
        "structure_and_topology",
        "alert_screening",
    }
    assert "minimol_solubility_log_s" in BUNDLE_PRIMITIVES["ionization_and_solubility"]
