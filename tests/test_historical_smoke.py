"""Smoke test of the historical pipeline's profiling stage (wraps tools/smoke_test_historical.py)."""
from __future__ import annotations

import subprocess
import sys


def test_profile_only_smoke(root):
    r = subprocess.run(
        [sys.executable, str(root / "tools" / "smoke_test_historical.py")], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "SMOKE TEST: PASS" in r.stdout
