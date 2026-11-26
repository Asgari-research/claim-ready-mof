"""Release integrity: the frozen V6 baseline must stay byte-identical.

These tests guard the revision rule "preserve frozen V6": the historical
analysis source and canonical publication figures must match the hashes
recorded in ``provenance/``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys


def _sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_validate_release_script_passes(root):
    r = subprocess.run([sys.executable, str(root / "tools" / "validate_release.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_historical_source_unchanged(root):
    recorded = json.loads((root / "provenance" / "source_hashes.json").read_text())
    for fname, meta in recorded.items():
        assert _sha256(root / "src" / fname) == meta["public_source_sha256"], f"src/{fname} was modified"


def test_canonical_figures_unchanged(root):
    with (root / "provenance" / "final_figure_sha256.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 15
    for row in rows:
        assert _sha256(root / row["file"]) == row["sha256"], row["file"]
