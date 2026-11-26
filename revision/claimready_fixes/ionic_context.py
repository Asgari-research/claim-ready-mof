"""Token-aware ionic-context detection (reviewer comment R1.2).

Historical behaviour (``src/`` ``row_flags``)::

    ionic = any(k in str(path).lower() for k in ['ion', 'charged', 'salt', 'ncr'])

This has two defects:

1. It is a substring test, so ``adsorption``, ``regression``, ``version`` and
   ``selection`` all count as ionic.
2. It tests the *whole absolute path*, so the result depends on where the data
   is installed. A data root under ``/home/<user>/Documents/Professional/...``
   ("profess-ion-al") marks every input file as ionic. This happened in the
   November 2026 reconstructed-input reruns: with that data root, all 278,492
   chemistry-trust rows were ionic, MOSAEC-DB lost its ``flagged_or_inconsistent``
   rows (SI Table S8: 200 -> 159 rows) and the representative rule cards
   changed (SI Table S15: 32 -> 24 rows). Moving the data root to a neutral
   folder restored both tables exactly (see revision/reproducibility/).

Here the path is split into alphanumeric tokens and a token must equal a
keyword exactly. Pass ``root`` (the data root) so that only the part of the
path *inside* the data root is considered.

``ncr`` is **not** an ionic keyword. In CoRE MOF, NCR means
"not computation-ready" (structures that failed the computation-ready checks),
which says nothing about framework charge. The historical rule treated it as
ionic; the corrected default does not.
"""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Iterable, List, Union

HISTORICAL_KEYWORDS = ("ion", "charged", "salt", "ncr")
DEFAULT_KEYWORDS = frozenset({"ion", "ions", "ionic", "charged", "salt", "salts"})

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


def relative_to_root(path: PathLike, root: PathLike | None) -> str:
    """Return ``path`` relative to ``root`` (``/`` separators); unchanged if ``root`` is None.

    Works for Windows and POSIX spellings alike, without touching the file system.
    """
    p = str(path).replace("\\", "/")
    if root is None:
        return p
    r = str(root).replace("\\", "/").rstrip("/") + "/"
    if p.lower().startswith(r.lower()):
        return p[len(r):]
    raise ValueError(f"{path!r} is not inside data root {root!r}")


def is_ionic_context(path: PathLike, keywords: Iterable[str] = DEFAULT_KEYWORDS,
                     root: PathLike | None = None) -> bool:
    """Return True only if a whole token of the path (inside ``root``, if given) equals a keyword."""
    kw = {k.lower() for k in keywords}
    return any(t in kw for t in path_tokens(relative_to_root(path, root)))


def compare_rules(paths: Iterable[PathLike], keywords: Iterable[str] = DEFAULT_KEYWORDS,
                  root: PathLike | None = None):
    """Return a DataFrame comparing historical vs token-aware decisions per path.

    Useful for the R1.2 sensitivity analysis: rows where the two rules disagree
    are exactly the files whose charge status would change.
    """
    import pandas as pd

    kw = frozenset(keywords)
    rows = []
    for p in paths:
        h = historical_is_ionic_context(p)
        n = is_ionic_context(p, kw, root=root)
        rows.append({"path": str(p), "historical_ionic": h, "token_ionic": n, "changed": h != n})
    return pd.DataFrame(rows, columns=["path", "historical_ionic", "token_ionic", "changed"])
