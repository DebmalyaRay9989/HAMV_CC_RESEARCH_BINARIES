# Frozen HAMV-CC Experimental Protocol

## Primary matrix

For the 638-document Archive-A case study, use the same source manifest, preprocessing, feature ordering, and seed list for every condition. The recommended primary seed list is `42,43,44,45,46,47,48,49,50,51`.

### Representation / component ablations

1. `full`: reliability weighting + missing-view masks + cross-view consistency + cluster consistency + balance.
2. `no_reliability`: replace learned reliability weighting with uniform available-view weights.
3. `no_missing_mask`: treat all views as available; retain this only as a deliberately incorrect robustness control.
4. `no_cross_view`: remove cross-view consistency term.
5. `no_cluster_consistency`: remove cluster-consistency term.
6. `fixed_k`: use the predeclared k rather than stability-aware selection.

The downstream K-Means implementation remains fixed inside this matrix.

## Missing-view matrix

Run matched seeds under:

- `full`
- `no_text`
- `no_visual`
- `no_structural`
- `only_text`
- `only_visual`
- `only_structural`

Do not create a zero-view sample. The loader rejects such cases.

## Statistical rules

- Report mean ± SD across matched seeds.
- Report 95% bootstrap confidence intervals for means.
- Compare ablations with paired seed differences.
- Report Wilcoxon signed-rank p-values when the paired sample permits it.
- Apply Benjamini–Hochberg correction over the declared family of paired tests.
- Report standardized paired effect size as mean paired difference / SD of paired differences.
- Retain every raw run, including failures.

## Anti-leakage rules

- Do not use human-audit labels to train, tune, select k, or choose a representation.
- Do not use RVL-CDIP test labels until all representation and k-selection decisions are frozen.
- Use the private archive only through a deterministic source manifest.
- If duplicate files or near-duplicates cross source splits, resolve them before confirmatory analysis.
