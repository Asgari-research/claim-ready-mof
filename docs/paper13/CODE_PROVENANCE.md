# Paper 13 code provenance and release scope

## Separation of scientific analysis and figure rendering

This repository separates the historical scientific-analysis source from the standalone final publication-figure regeneration layer.

The final renderer under `reproduce/figures/paper13/` operates on frozen saved figure-source tables. It does not refit machine-learning models, regenerate descriptors, rerun simulations, or alter chemistry/evidence classification.

The historical scientific-analysis source should be preserved without structural refactoring unless a separate behavior-validation task is undertaken. The compact release materials do not by themselves establish the exact historical software environment or a complete end-to-end rerun of every upstream data resource.

## Development assistance

The final publication-figure scripts were developed/refactored with generative-AI assistance during the publication-preparation workflow and then repeatedly executed and checked against frozen source tables. Scientific interpretation, source selection, numerical verification, and responsibility for the final repository content remain with the authors.

No claim is made that the historical scientific-analysis code was or was not AI-assisted solely on the basis of source-text scans. Absence of assistant/product names in source files is not evidence of authorship history.

## Preservation principle

Repository cleanup must not silently change scientific function order, numerical constants, model settings, split logic, identifier handling, chemistry rules, or archived result values. Any future scientific-code correction should be isolated, documented, and accompanied by an explicit validation/recomputation plan rather than retroactively attributing unchanged outputs to modified code.
