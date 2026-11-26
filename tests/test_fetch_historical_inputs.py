"""Offline tests for tools/fetch_historical_inputs.py (no downloads)."""
from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_historical_inputs", ROOT / "tools" / "fetch_historical_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_lists_the_509_historical_files(fetch):
    rows = fetch.load_manifest(fetch.DEFAULT_MANIFEST)
    assert len(rows) == 509
    assert all("\\" not in r["rel"] for r in rows)


def test_every_source_is_pinned(fetch):
    for src in fetch.SOURCES:
        assert src["kind"] in {"zenodo", "figshare", "url"}
        if src["kind"] == "url":
            assert len(src["md5"]) == 32 and src["size"] > 0
            assert "/raw/" in src["url"] and len(src["url"].split("/raw/")[1].split("/")[0]) == 40  # commit hash


def test_names_match_across_case_dashes_and_spaces(fetch):
    assert fetch.norm("landfill-CO2.csv") == fetch.norm("landfill_co2.csv")
    assert fetch.norm("Framework details.csv") == fetch.norm("framework_details.csv")


def test_best_candidate_prefers_matching_size_then_path(fetch, tmp_path):
    a = tmp_path / "x" / "other" / "f.csv"
    b = tmp_path / "x" / "core" / "f.csv"
    c = tmp_path / "x" / "core" / "sub" / "f.csv"
    for p, n in [(a, 10), (b, 10), (c, 7)]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * n)
    chosen, size, ambiguous = fetch.best_candidate("data_raw/x/core/f.csv", 10, [a, b, c])
    assert chosen == b and size == 10 and not ambiguous


def test_place_all_reports_ok_missing_and_not_public(fetch, tmp_path):
    staging = tmp_path / "staging"
    (staging / "qmof").mkdir(parents=True)
    (staging / "qmof" / "QMOF-Props.csv").write_bytes(b"abc")
    manifest = [
        {"rel": "data_raw/qmof/qmof_props.csv", "size": 3},
        {"rel": "data_raw/qmof/missing.csv", "size": 1},
        {"rel": "PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt", "size": 5},
    ]
    report = {r["relative_path"]: r["status"] for r in fetch.place_all(manifest, staging, tmp_path / "out")}
    assert report == {
        "data_raw/qmof/qmof_props.csv": "OK",
        "data_raw/qmof/missing.csv": "MISSING",
        "PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt": "NOT_PUBLIC",
    }
    assert (tmp_path / "out" / "data_raw" / "qmof" / "qmof_props.csv").read_bytes() == b"abc"


def test_unsafe_zip_member_is_refused(fetch, tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError):
        fetch.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_tar_extracts_inside_destination(fetch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("ok")
    arc = tmp_path / "a.tar.gz"
    with tarfile.open(arc, "w:gz") as t:
        t.add(f, arcname="dir/a.txt")
    fetch.safe_extract(arc, tmp_path / "out")
    assert (tmp_path / "out" / "dir" / "a.txt").read_text() == "ok"
