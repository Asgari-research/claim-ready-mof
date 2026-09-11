# From Computation-Ready to Claim-Ready: Evidence-Aware Data for MOF Machine Learning

This repository accompanies the manuscript **“From Computation-Ready to Claim-Ready: Evidence-Aware Data for MOF Machine Learning.”** It provides the preserved scientific analysis source, final publication figures, frozen figure-source tables, and a standalone figure-regeneration workflow.

## Scope

The study examines how evidence retained in digital metal–organic framework (MOF) resources constrains the claims that can responsibly be supported by machine-learning analyses. It combines resource-level profiling, chemistry/evidence observability diagnostics, identifier/join accounting, and retrospective ML stress testing.

The repository intentionally distinguishes **historical scientific analysis** from **final publication rendering**:

- `src/` — preserved scientific-analysis source associated with the completed historical run.
- `reproduce/figures/` — standalone renderer using frozen derived source tables only.
- `figures/` — canonical final publication PDFs.
- `docs/` — reproducibility, provenance, figure mapping, and scientific-boundary documentation.
- `provenance/` — sanitized run metadata and source hashes.

## Quick start: reproduce the publication figures

```bash
conda env create -f reproduce/figures/environment.yml
conda activate claim_ready_mof_figures
python reproduce/figures/run_figures.py
```

Final manuscript rendering uses Arial from a lawful local installation. On Linux/WSL:

```bash
export CLAIM_READY_MOF_ARIAL_DIR="$HOME/.local/share/fonts/Arial"
python reproduce/figures/run_figures.py
```

The renderer writes regenerated assets under `reproduce/figures/outputs/` and does **not** overwrite the canonical PDFs under `figures/`.

## Historical scientific analysis

The original full analysis ran against a large local collection of third-party MOF resources and completed all recorded pipeline stages successfully. Raw third-party databases are not redistributed here. The exact historical run configuration, captured software versions, run summary, and limitations are documented in `docs/HISTORICAL_RUN.md`.

The two scripts in `src/` preserve the executable behavior of the supplied run-source files. Public-portability edits are limited to module-level documentation text; normalized executable AST hashes are unchanged. See `docs/CODE_PROVENANCE.md`.

## Final figure set

- Main Figures **1–5**
- Supporting Information Figures **S0–S9**

The final PDFs are under `figures/main/` and `figures/supplementary/`. Figure-to-source mapping is provided in `docs/FIGURE_TO_DATA_MAP.csv`.

## Reproducibility boundary

The repository supports transparent inspection of the scientific workflow and direct regeneration of the final figure layer. It is **not** a redistribution of every upstream third-party database used in the historical analysis. Claims that depend on unavailable upstream resources should therefore be interpreted together with the manuscript, the documented data provenance, and the original providers’ terms.

Operational evidence/validation categories are pipeline constructs. Composite readiness scores are reporting constructs, not calibrated probabilities of chemical correctness. Displayed ML examples are retrospective, not prospective validation.

## Citation

See `CITATION.cff` and `AUTHORS.md`. Publication DOI/journal metadata can be added to `CITATION.cff` once assigned.

## License

Repository software and documentation are released under the MIT License in `LICENSE`. Raw third-party MOF databases are not redistributed or relicensed by this repository.
