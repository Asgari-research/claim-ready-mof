"""Editable visual settings for Paper 13 figure redesign.

No scientific values live here. This file controls typography, dimensions and colors only.
"""
from pathlib import Path

FULL_WIDTH_IN = 7.20  # ~183 mm
PNG_DPI = 600
FONT_FAMILY = "Arial"
FONT_SIZES = {
    "base": 9.2,
    "axis": 9.4,
    "tick": 8.6,
    "title": 10.2,
    "panel": 12.5,
    "legend": 8.4,
    "annotation": 8.5,
    "small": 8.0,
}

RESOURCE_COLORS = {
    "MOSAEC-DB": "#009E73",
    "CoRE MOF 2024": "#0072B2",
    "CoRE MOF 2025 metadata": "#56B4E9",
    "ARC-MOF": "#CC79A7",
    "QMOF": "#E69F00",
    "CSD-derived context": "#D55E00",
    "Other/unknown": "#999999",
}
SHORT = {
    "MOSAEC-DB": "MOSAEC",
    "CoRE MOF 2024": "CoRE-24",
    "CoRE MOF 2025 metadata": "CoRE-25",
    "ARC-MOF": "ARC-MOF",
    "QMOF": "QMOF",
    "CSD-derived context": "CSD ctx",
    "Other/unknown": "Other",
}
TRUST_COLORS = {
    "validated_observable": "#009E73",
    "validated_not_observable": "#56B4E9",
    "uncertain_observable": "#E69F00",
    "uncertain_unobservable": "#999999",
    "flagged_or_inconsistent": "#D55E00",
    "not_scanned": "#ECECEC",
}
SPLIT_COLORS = {"random": "#0072B2", "descriptor_grouped": "#D55E00"}
NEUTRAL = "#606060"
GRID = "#D9D9D9"
