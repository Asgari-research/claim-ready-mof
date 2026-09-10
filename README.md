# From Computation-Ready to Claim-Ready: Evidence-Aware Data for MOF Machine Learning

This repository accompanies a study of how the evidence retained in digital metal–organic framework (MOF) data constrains the scientific claims that can be made from machine-learning analyses.

The project combines resource-level data profiling, representation-level chemistry/evidence diagnostics, identifier-join accounting, and a retrospective machine-learning stress test. The emphasis is on **claim scope, evidence traceability, and reporting boundaries**, rather than ranking one MOF database as universally superior.

## Scientific scope

The archived chemistry/evidence regimes are operational categories produced by the historical analysis pipeline. Some validation state can inherit source/curation assumptions and therefore should not be interpreted as independent row-level chemical validation. Composite readiness scores are reporting constructs, not calibrated probabilities of chemical correctness. The displayed ML examples are retrospective and include performance-based selection.

Accordingly, this repository supports an **evidence-focused MOF data study with a retrospective ML illustration**. It should not be read as a prospectively validated chemical-error detector, universal database ranking, or prospective discovery benchmark.

## Final publication figures

The finalized figure architecture is:

- **Main Figures 1–5**
- **Supporting Information Figures S0–S9**

The former main decision-framework Figure 6 is now **SI Figure S9**.

Final PDF artwork is stored under:

```text
figures/main/
figures/supplementary/
```

The publication figure renderer is maintained under:

```text
reproduce/figures/paper13/
```

It regenerates artwork from frozen saved figure-source tables. It does **not** refit models, regenerate descriptors, rerun simulations, or reclassify chemistry/evidence states.

## Repository layout

```text
README.md
LICENSE
figures/
  main/                         final Main Figures 1–5
  supplementary/                final SI Figures S0–S9
reproduce/figures/paper13/       publication figure renderer and QA
docs/paper13/                    figure maps, provenance, and scientific boundaries
src/                             historical scientific analysis source (added/reviewed separately)
```

The scientific-analysis source and the publication-figure renderer are intentionally treated as separate layers. This avoids presenting the newer figure-layout code as if it were the code that generated every historical scientific result.

## Figure regeneration

The renderer uses Python 3 and standard scientific-Python packages listed in:

```text
reproduce/figures/paper13/requirements.txt
```

On Linux/WSL, once the reviewed figure-source tables are available in the expected `data/` directory:

```bash
cd reproduce/figures/paper13
python -m pip install -r requirements.txt
export PAPER13_ARIAL_DIR="$HOME/.local/share/fonts/Arial"
bash RUN_ALL_WSL.sh
```

Arial itself is **not distributed** in this repository. Users must provide a legally installed local copy or adapt the renderer to an available font for non-publication testing.

## Reproducibility boundary

The compact publication release is not presented as a complete end-to-end reconstruction of every historical upstream step. In particular, the original scientific environment, all raw third-party resources, and every historical intermediate product are not currently represented by the compact release materials.

The repository therefore distinguishes:

1. historical scientific-analysis source;
2. frozen publication-level result/figure inputs;
3. final figure-regeneration code and artwork.

See `docs/paper13/DATA_AND_REPRODUCIBILITY.md` and `docs/paper13/SCIENTIFIC_INTEGRITY_BOUNDARIES.md` for the detailed scope.

## Data and third-party resources

Raw third-party MOF databases, CIF archives, proprietary fonts, credentials, and machine-specific paths are not intended for public redistribution here.

Derived figure-source tables are added only after their redistribution status and portability have been reviewed. Sanitizing historical local paths is a portability/privacy edit and does not itself establish a redistribution licence.

## Code provenance and development assistance

The final publication-figure scripts were developed/refactored with generative-AI assistance and were subsequently run and checked against frozen saved source tables. Scientific interpretation, source selection, numerical verification, and responsibility for the final repository content remain with the authors.

The historical scientific-analysis code is preserved separately and is not structurally refactored merely for cosmetic reasons. See `docs/paper13/CODE_PROVENANCE.md`.

## Licence

The repository currently carries the **MIT License** for repository software and documentation, as recorded in the repository's existing `LICENSE` file.

Third-party datasets and derived materials may remain subject to their original providers' terms. The MIT License should not be interpreted as relicensing third-party data.

## Citation

Final manuscript and software citation metadata should be added when the publication details are fixed. Until then, users should cite the original data/resources according to their provider requirements and cite the associated manuscript when available.
