"""Characterisation tests: document reviewer-flagged behaviour of the FROZEN pipeline.

Each test asserts what the historical code in ``src/`` actually does today, so
the evidence behind each reviewer response is executable rather than claimed.
They are expected to PASS while ``src/`` is frozen. If someone edits ``src/``
and one starts failing, the historical baseline has changed and the change
must be logged before any result is reused.

Reviewer IDs refer to DD-ART-10-2026-000896 (see revision/reviewer_matrix.md).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# R1.2 ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "ARC-MOF/adsorption/CO2_uptake.csv",
        "CoRE2019/regression_targets.csv",
        "QMOF/version2/props.csv",
        "features/selection/descriptors.csv",
    ],
)
def test_R1_2_ionic_flag_is_path_substring(historical, path):
    """'ion' inside ordinary words marks the whole file as an ionic context."""
    src = inspect.getsource(historical.row_flags)
    assert "any(k in str(p).lower() for k in ['ion','charged','salt','ncr'])" in src
    assert any(k in path.lower() for k in ["ion", "charged", "salt", "ncr"])


@pytest.mark.parametrize(
    "root,expected_role",
    [
        ("C:/Users/someone/mof", "general_metadata"),
        ("/home/someone/crmof_data", "general_metadata"),
        ("/home/someone/Documents/Professional/MOFML/Data", "curation_or_validation"),
    ],
)
def test_R1_2_file_role_depends_on_install_folder(historical, root, expected_role):
    """role_of() also searches the absolute path, so the folder the data sits in changes the role.

    Observed in the November 2026 reruns: with a data root containing "Professional", about 7,240
    of 22,016 rows in SI Table S2 changed role to curation_or_validation.
    """
    rel = "data_raw/arc_mof/clean_data/clean_data.csv"
    assert historical.role_of(Path(f"{root}/{rel}")) == expected_role


def test_R1_2_ionic_flag_reads_absolute_path(historical):
    """row_flags() applies the substring rule to str(p), the full path of the input file."""
    src = inspect.getsource(historical.row_flags)
    assert "ionic=any(k in str(p).lower()" in src
    assert "out['file_path']=str(p)" in src


# R1.4 / R2.6 -------------------------------------------------------------
def test_R1_4_join_drops_second_pressure_state(historical):
    tgt = pd.DataFrame(
        {"name": ["MOF_A", "MOF_A", "MOF_B", "MOF_B"], "pressure_bar": [0.15, 0.015, 0.15, 0.015], "uptake": [2.0, 0.4, 3.1, 0.6]}
    )
    desc = pd.DataFrame({"name": ["MOF_A", "MOF_B"], "pld": [4.0, 5.0]})
    td, _, _ = historical.prepare_variant_joined_table(tgt, desc, "name", "name")
    assert len(td) == 2, "one row per material survives; the other condition is lost"
    td_rev, _, _ = historical.prepare_variant_joined_table(tgt.iloc[[1, 0, 3, 2]], desc, "name", "name")
    assert sorted(td["uptake"]) != sorted(td_rev["uptake"]), "kept label depends on row order"


# R2.2 --------------------------------------------------------------------
def test_R2_2_sym_stripped_matches_a_different_structure(historical):
    tgt = pd.DataFrame({"name": ["str_m1_o1_o1_pcu_sym.10"], "y": [2.0]})
    desc = pd.DataFrame({"name": ["str_m1_o1_o1_pcu_sym.11", "str_m1_o1_o1_pcu_sym.10"], "pld": [5.0, 5.1]})
    _, dd, _ = historical.prepare_variant_joined_table(tgt, desc, "name", "name", "sym_stripped")
    assert dd["name"].tolist() == ["str_m1_o1_o1_pcu_sym.11"], "sym.10 target receives sym.11 descriptors"


# R1.5 / R2.4 -------------------------------------------------------------
def test_R1_5_quantile_groups_split_tied_values(historical):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pld": np.round(rng.uniform(3, 8, 2000), 1)})
    groups, name = historical.choose_group_series_for_joined(df, ["pld"], [])
    assert name == "pld_quantile_group"
    n_split = int((pd.crosstab(df.pld, groups) > 0).sum(axis=1).gt(1).sum())
    assert n_split > 0, "identical descriptor values fall in different groups"


# R2.3 --------------------------------------------------------------------
def test_R2_3_feature_screening_uses_all_rows_before_split(historical):
    src = inspect.getsource(historical.xy)
    assert "safe_missing_fraction(s)>.65" in src and "safe_nunique(s)<=1" in src
    assert "train_test_split" not in src and ".split(" not in src


# R1.3 --------------------------------------------------------------------
def test_R1_3_each_descriptor_family_joined_separately(historical):
    src = inspect.getsource(historical.step_descriptor_joined_ml)
    assert "for _,dr in desc_rows.iterrows():" in src
    assert "best_identifier_join(target_df, desc_df, cfg)" in src


# Maintainability -----------------------------------------------------------
def test_many_functions_are_redefined(root):
    import ast

    tree = ast.parse((root / "src" / "chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py").read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    redefined = {n for n in names if names.count(n) > 1}
    assert len(redefined) >= 50, "only the LAST definition of each is live; see revision/logs/live_definitions.md"
