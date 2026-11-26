"""Tests for tools/compare_reproducibility.py on small synthetic tables."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cmp():
    spec = importlib.util.spec_from_file_location("compare_reproducibility", ROOT / "tools" / "compare_reproducibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "raw",
    [
        r"C:\Users\someone\mof\data_raw\arc_mof\clean_data.csv",
        "/home/someone/crmof_data/data_raw/arc_mof/clean_data.csv",
        "<DATA_ROOT>\\data_raw\\arc_mof\\clean_data.csv",
    ],
)
def test_paths_are_normalised_to_data_root(cmp, raw):
    assert cmp.norm_text(raw) == "<DATA_ROOT>/data_raw/arc_mof/clean_data.csv"


def test_identical_tables_in_different_row_order(cmp):
    a = pd.DataFrame({"resource": ["A", "B", "C"], "n": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
    b = a.iloc[::-1].reset_index(drop=True)
    res, diffs = cmp.compare_table("Figure_4c_score_sensitivity", a, b)
    assert res["status"] == "IDENTICAL"
    assert res["alignment"] == "keys:resource"
    assert diffs.empty


def test_small_float_noise_is_identical_and_larger_change_is_close(cmp):
    a = pd.DataFrame({"resource": [f"r{i}" for i in range(200)], "x": np.linspace(1, 2, 200)})
    b = a.copy()
    b["x"] = b["x"] * (1 + 1e-9)
    assert cmp.compare_table("t", a, b)[0]["status"] == "IDENTICAL"
    b.loc[0, "x"] *= 1.0005
    res, diffs = cmp.compare_table("t", a, b)
    assert res["status"] == "CLOSE"
    assert len(diffs) == 1 and diffs.iloc[0]["resource"] == "r0"


def test_missing_rows_are_reported(cmp):
    a = pd.DataFrame({"resource": ["A", "B", "C", "D"], "n": [1, 2, 3, 4]})
    b = a[a.resource != "D"]
    res, _ = cmp.compare_table("t", a, b)
    assert res["status"] == "DIFFERENT"
    assert res["rows_only_archived"] == 1


def test_timing_columns_are_ignored(cmp):
    a = pd.DataFrame({"relative_path": ["x.csv", "y.csv"], "size_bytes": [10, 20], "read_seconds": [1.0, 2.0]})
    b = a.assign(read_seconds=[5.0, 9.0])
    res, _ = cmp.compare_table("SI_Table_S1_complete_file_level_profile", a, b)
    assert res["status"] == "IDENTICAL"
    assert res["ignored_volatile_cols"] == "read_seconds"


def test_key_hint_keeps_rows_aligned_when_a_label_changes(cmp):
    a = pd.DataFrame({"relative_path": ["f.csv", "f.csv"], "column_name": ["a", "b"],
                      "role": ["general_metadata"] * 2, "missing_count": [0, 1]})
    b = a.assign(role=["curation_or_validation"] * 2)
    res, diffs = cmp.compare_table("SI_Table_S2_complete_column_level_profile", a, b)
    assert res["alignment"] == "keys:relative_path,column_name"
    assert res["rows_only_archived"] == 0
    assert set(diffs["kind"]) == {"text differs: role"}


def test_headline_table_is_rebuilt_from_s6(cmp):
    s6 = pd.DataFrame({
        "source_table": ["T"] * 4, "target_column": ["y"] * 4, "feature_source": ["f"] * 4,
        "descriptor_family": ["d"] * 4, "model": ["hgb", "hgb", "dummy", "dummy"],
        "split_type": ["random"] * 4, "repeat": [0, 1, 0, 1], "rmse": [1.0, 3.0, 5.0, 5.0], "r2": [0.5, 0.7, 0.0, 0.0],
    })
    archived = pd.DataFrame({"source_table": ["T"], "target_column": ["y"], "feature_source": ["f"],
                             "descriptor_family": ["d"], "model": ["hgb"], "split_type": ["random"],
                             "n_repeats": [2], "mean_rmse": [2.0], "sd_rmse": [np.sqrt(2)], "mean_r2": [0.6]})
    derived = cmp.derive_headline(archived, s6)
    res, _ = cmp.compare_table(cmp.HEADLINE, archived, derived)
    assert len(derived) == 1
    assert res["status"] == "IDENTICAL"


def test_end_to_end_report(cmp, tmp_path):
    archive = tmp_path / "archive" / "si"
    run = tmp_path / "run" / "tables" / "si"
    archive.mkdir(parents=True)
    run.mkdir(parents=True)
    t = pd.DataFrame({"resource": ["A", "B"], "n": [1, 2]})
    t.to_csv(archive / "Table_x.csv", index=False)
    t.to_csv(run / "Table_x.csv", index=False)
    pd.DataFrame({"resource": ["A"], "n": [1]}).to_csv(archive / "SI_Table_S0b_inventory_folder_summary.csv", index=False)
    assert cmp.main(["--rerun", str(tmp_path / "run"), "--archive", str(tmp_path / "archive")]) == 0
    summary = pd.read_csv(tmp_path / "run" / "reproducibility" / "reproducibility_summary.csv")
    status = dict(zip(summary.table, summary.status))
    assert status["Table_x"] == "IDENTICAL"
    assert status["SI_Table_S0b_inventory_folder_summary"].startswith("BLOCKED")
    assert (tmp_path / "run" / "reproducibility" / "REPRODUCIBILITY_REPORT.md").exists()
