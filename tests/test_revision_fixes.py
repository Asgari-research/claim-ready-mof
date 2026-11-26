"""Tests for the corrected helpers in revision/claimready_fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from claimready_fixes import condition_keys as ck
from claimready_fixes import ionic_context as ic
from claimready_fixes import screening as sc
from claimready_fixes import splits as sp


# ionic_context -------------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("ARC-MOF/adsorption/CO2_uptake.csv", False),
        ("CoRE2019/regression_targets.csv", False),
        ("QMOF/version2/props.csv", False),
        ("features/selection/descriptors.csv", False),
        ("CoRE2024/ION/structures.csv", True),
        ("charged_frameworks/table.csv", True),
        ("data/ionic-mofs.csv", True),
        ("CoRE/NCR_subset.csv", False),  # NCR = not computation-ready, not ionic
        ("data_raw/core_mof_2024/NCR_ION_SI.xlsx", True),  # ION token, not NCR
        ("data_raw/core_mof_2024/NCR_ASR_SI.xlsx", False),
        ("data_raw/core_mof_2024/unmodified_check_for_NCR_SI.xlsx", False),
        ("IonCharged/x.csv", True),
    ],
)
def test_token_aware_ionic_context(path, expected):
    assert ic.is_ionic_context(path) is expected


def test_ncr_is_not_an_ionic_keyword():
    assert "ncr" not in ic.DEFAULT_KEYWORDS
    assert "ncr" in ic.HISTORICAL_KEYWORDS
    assert ic.historical_is_ionic_context("CoRE/NCR_subset.csv") is True
    assert ic.is_ionic_context("CoRE/NCR_subset.csv") is False


INSTALL_ROOTS = [
    "C:\\Users\\someone\\mof",
    "/home/someone/Documents/Professional/Jobs/MOFML/Data",
    "/home/someone/crmof_data",
    "/Users/someone/Desktop/regression_project",
]


@pytest.mark.parametrize("root", INSTALL_ROOTS)
@pytest.mark.parametrize(
    "rel,expected",
    [
        ("data_raw/mosaec_db/database/mosaec_db.csv", False),
        ("data_raw/core_mof_2024/ION_data_SI_20250204.csv", True),
        ("data_raw/csd_mof_collection/CSD_MOF_Collection/Charged frameworks.csv", True),
    ],
)
def test_ionic_context_does_not_depend_on_install_folder(root, rel, expected):
    sep = "\\" if root.startswith("C:") else "/"
    full = root + sep + rel.replace("/", sep)
    assert ic.is_ionic_context(full, root=root) is expected


def test_historical_rule_depends_on_install_folder():
    rel = "data_raw/mosaec_db/database/mosaec_db.csv"
    assert ic.historical_is_ionic_context("C:/Users/someone/mof/" + rel) is False
    assert ic.historical_is_ionic_context("/home/someone/Documents/Professional/MOFML/Data/" + rel) is True


def test_relative_to_root_rejects_outside_paths():
    with pytest.raises(ValueError):
        ic.relative_to_root("/elsewhere/x.csv", "/home/someone/crmof_data")


def test_compare_rules_flags_changed_files():
    df = ic.compare_rules(["a/adsorption.csv", "b/ION/x.csv"])
    assert df["changed"].tolist() == [True, False]


# condition_keys ------------------------------------------------------------
@pytest.fixture
def two_pressures():
    t = pd.DataFrame(
        {
            "name": ["MOF_A", "MOF_A", "MOF_B", "MOF_B"],
            "gas": ["CO2"] * 4,
            "temperature_K": [298] * 4,
            "pressure_bar": [0.15, 0.015, 0.15, 0.015],
            "uptake": [2.0, 0.4, 3.1, 0.6],
        }
    )
    d = pd.DataFrame({"name": ["mof_a.cif", "MOF_B"], "pld": [4.0, 5.0]})
    return t, d


def test_canonical_key_keeps_conditions_distinct(two_pressures):
    t, _ = two_pressures
    k = ck.build_canonical_key(t, "name", resource="ARC-MOF")
    assert k["canonical_key"].nunique() == 4
    assert k["canonical_key"].iloc[0] == "ARC-MOF|mof_a|co2|298|0.15"
    assert ck.audit_key_collisions(k, "canonical_key", "uptake").empty


def test_canonical_key_requires_condition_columns(two_pressures):
    t, _ = two_pressures
    with pytest.raises(KeyError):
        ck.build_canonical_key(t.drop(columns=["pressure_bar"]), "name")


def test_audit_reports_conflicting_targets(two_pressures):
    t, _ = two_pressures
    k = ck.build_canonical_key(t, "name", condition_cols=())  # id-only key: the historical defect
    audit = ck.audit_key_collisions(k, "canonical_key", "uptake")
    assert len(audit) == 2 and audit["targets_conflict"].all()


def test_condition_preserving_join_keeps_every_observation(two_pressures):
    t, d = two_pressures
    joined, rep = ck.condition_preserving_join(t, d, "name", "name")
    assert len(joined) == 4 and rep.n_targets_unmatched == 0
    assert sorted(joined["uptake"]) == [0.4, 0.6, 2.0, 3.1]
    # Row order no longer changes the labels.
    joined_rev, _ = ck.condition_preserving_join(t.iloc[[1, 0, 3, 2]], d, "name", "name")
    assert sorted(joined_rev["uptake"]) == sorted(joined["uptake"])


def test_join_refuses_ambiguous_descriptors(two_pressures):
    t, d = two_pressures
    d2 = pd.concat([d, pd.DataFrame({"name": ["MOF_A"], "pld": [9.9]})], ignore_index=True)
    with pytest.raises(ck.AmbiguousDescriptorError):
        ck.condition_preserving_join(t, d2, "name", "name")
    joined, rep = ck.condition_preserving_join(t, d2, "name", "name", on_ambiguous="drop")
    assert rep.ambiguous_descriptor_ids == ["mof_a"] and set(joined["name"]) == {"MOF_B"}


def test_exact_id_does_not_merge_different_structures():
    s = pd.Series(["str_m1_o1_o1_pcu_sym.10", "str_m1_o1_o1_pcu_sym.11", "ABAVIJ", "ABAVIJ01"])
    assert ck.normalize_exact_id(s).nunique() == 4


# splits --------------------------------------------------------------------
def test_tie_safe_groups_never_split_ties():
    rng = np.random.default_rng(0)
    v = pd.Series(np.round(rng.uniform(3, 8, 2000), 1))
    g = sp.quantile_groups_tie_safe(v, 10)
    assert sp.tied_values_split_across_groups(v, g) == 0
    sizes = g.value_counts()
    assert 8 <= len(sizes) <= 10 and sizes.min() > 100


def test_tie_safe_groups_handle_missing():
    g = sp.quantile_groups_tie_safe(pd.Series([1.0, None, 2.0, 3.0]), 2)
    assert g.isna().tolist() == [False, True, False, False]


def test_group_overlap_and_duplicate_rows():
    assert sp.group_overlap(pd.Series([1, 2]), pd.Series([2, 3])) == {2}
    a = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    b = pd.DataFrame({"x": [2.0, 9.0], "y": [4.0 + 1e-12, 9.0]})
    assert sp.exact_duplicate_rows_across_split(a, b) == 1


# screening -----------------------------------------------------------------
def test_screen_is_learned_from_training_rows_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, np.nan, np.nan, 1.0], "c": [5.0] * 4})
    test = pd.DataFrame({"a": [5.0, 6.0], "b": [1.0, 2.0], "c": [1.0, 2.0]})
    s = sc.TrainOnlyFeatureScreen().fit(train)
    assert s.kept_columns_ == ["a"]  # b too missing, c constant -- in TRAIN
    assert list(s.transform(test).columns) == ["a"]


def test_pipeline_runs_inside_cross_validation():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    X.loc[:90, "f"] = np.nan  # mostly missing
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=120)
    pipe = sc.make_train_only_pipeline(Ridge())
    scores = cross_val_score(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    assert scores.mean() > 0.9
