"""Token-aware ionic-context detection (reviewer comment R1.2).

Historical behaviour (``src/`` ``row_flags``)::

    ionic = any(k in str(path).lower() for k in ['ion', 'charged', 'salt', 'ncr'])

This is a substring test on the whole path, so ``adsorption``, ``regression``,
``version`` and ``selection`` all count as ionic. Here the path is split into
alphanumeric tokens and a token must equal a keyword exactly.

The default keyword set keeps the historical keywords (plus plural forms) so
that the only behavioural change is the substring defect. Whether ``ncr``
should indicate an ionic context at all is an open scientific question and is
left to the authors (see revision/logs/U0_feasibility.md).
"""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Iterable, List, Union

HISTORICAL_KEYWORDS = ("ion", "charged", "salt", "ncr")
DEFAULT_KEYWORDS = frozenset({"ion", "ions", "ionic", "charged", "salt", "salts", "ncr"})

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")

PathLike = Union[str, PurePath]


def path_tokens(path: PathLike) -> List[str]:
    """Split a path into lower-case alphanumeric tokens.

    CamelCase boundaries are split too, so ``IonCharged`` -> ``ion``, ``charged``.
    """
    text = _CAMEL.sub("_", str(path))
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


def historical_is_ionic_context(path: PathLike) -> bool:
    """Exact re-implementation of the historical substring rule, for comparison."""
    p = str(path).lower()
    return any(k in p for k in HISTORICAL_KEYWORDS)


def is_ionic_context(path: PathLike, keywords: Iterable[str] = DEFAULT_KEYWORDS) -> bool:
    """Return True only if a whole path token equals an ionic keyword."""
    kw = {k.lower() for k in keywords}
    return any(t in kw for t in path_tokens(path))


def compare_rules(paths: Iterable[PathLike], keywords: Iterable[str] = DEFAULT_KEYWORDS):
    """Return a DataFrame comparing historical vs token-aware decisions per path.

    Useful for the R1.2 sensitivity analysis: rows where the two rules disagree
    are exactly the files whose charge status would change.
    """
    import pandas as pd

    kw = frozenset(keywords)
    rows = []
    for p in paths:
        h = historical_is_ionic_context(p)
        n = is_ionic_context(p, kw)
        rows.append({"path": str(p), "historical_ionic": h, "token_ionic": n, "changed": h != n})
    return pd.DataFrame(rows, columns=["path", "historical_ionic", "token_ionic", "changed"])
