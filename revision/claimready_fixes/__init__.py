"""Corrected, independently testable helpers for the Digital Discovery revision.

These modules do NOT modify the frozen historical pipeline under ``src/``.
They provide reviewed replacements for specific behaviours that reviewers
questioned, so that corrected analyses can be built and compared against the
historical baseline:

- ``ionic_context``   token-aware ionic-context detection (R1.2)
- ``condition_keys``  condition-preserving identity keys, collision audit and
                      a join that never silently drops rows (R1.4, R2.2, R2.6)
- ``splits``          tie-safe quantile groups and split-leakage checks
                      (R1.5, R2.4)
- ``screening``       train-only feature screening inside an sklearn
                      pipeline (R2.3)

Scientific choices (token lists, condition columns, thresholds) are explicit
parameters. Defaults mirror the historical settings unless a default was the
defect itself; changing them is a joint scientific decision, not a coding one.
"""

__all__ = ["ionic_context", "condition_keys", "splits", "screening"]
