"""Shared pytest fixtures.

``historical`` imports the frozen pipeline in ``src/`` as a module (its
``main()`` only runs under ``__main__``), so its live functions can be tested
directly without editing the file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SRC = ROOT / "src" / "chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py"

# Make ``revision/claimready_fixes`` importable as ``claimready_fixes``.
sys.path.insert(0, str(ROOT / "revision"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def historical():
    name = "historical_pipeline"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HISTORICAL_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for @dataclass on Python >= 3.12
    spec.loader.exec_module(mod)
    return mod
