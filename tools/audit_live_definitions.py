#!/usr/bin/env python3
"""List functions defined more than once in a Python file and which definition is live.

Python keeps only the last top-level definition of a name, so in the
historical pipeline earlier copies are dead code. Run before editing or
reviewing any function so you read the version that actually executes.

Usage:
    python tools/audit_live_definitions.py [path/to/file.py] [--out report.md]
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "src" / "chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py"


def audit(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name].append((node.lineno, node.end_lineno))
    return defs


def render(path: Path, defs) -> str:
    multi = {k: v for k, v in defs.items() if len(v) > 1}
    total_lines = len(path.read_text(encoding="utf-8").splitlines())
    dead = sum(e - s + 1 for v in multi.values() for s, e in v[:-1])
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path.name
    out = [
        f"# Live-definition audit: `{shown}`",
        "",
        f"- Top-level definitions: {len(defs)} names",
        f"- Names defined more than once: {len(multi)}",
        f"- Lines in superseded (dead) definitions: {dead} of {total_lines}",
        "",
        "Only the **live** (last) definition executes. Edit/review that one.",
        "",
        "| name | copies | live definition (lines) | superseded copies (start lines) |",
        "|---|---:|---|---|",
    ]
    for name, spans in sorted(multi.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        s, e = spans[-1]
        out.append(f"| `{name}` | {len(spans)} | {s}-{e} | {', '.join(str(a) for a, _ in spans[:-1])} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DEFAULT))
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.path)
    text = render(p, audit(p))
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
