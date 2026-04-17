# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `nb_fit()` / `nb_test()`: GPU-accelerated negative binomial GLM fitting with quasi-likelihood dispersion shrinkage (glmGamPoi approach).
- `nb_de()`: One-shot convenience wrapper around `nb_fit()` + `nb_test()`.
- `build_design()`: Shared design matrix builder with R-style formula support via patsy.
- `grouped()`: Per-group DE wrapper with cross-group multiple testing correction.
- `auroc()`: Pairwise AUROC effect size computation (now exported).
- Formula interface for `de()`, `nb_fit()`, and `nb_de()` via `formula=` parameter.
- `de()` now returns `log2fc` and `stat` columns alongside `coef`, `pval`, `padj`.

### Changed

- `de()` no longer loops over pairwise comparisons. It builds a single design matrix and tests one contrast per call. Use `grouped()` for per-group analysis.
- `de()` signature simplified: removed `mode`, `min_samples`, `log2fc_threshold` parameters.
- ANOVA backend uses closed-form F-test via Frisch-Waugh-Lovell projection (no iterative fitting).
- AUROC in `_effects.py` uses Mann-Whitney U rank-sum statistic instead of sort+trapezoid.
- `rank_de()` p-values computed in JAX instead of SciPy.

### Removed

- `de_enrichment_analysis()`: removed from public API.
- Pairwise comparison modes (`all_vs_ref`, `all_vs_all`, `1_vs_1`) from `de()`. These modes remain available in `log2fc()` and `auroc()`.
- cuML and PyDESeq2 backend references from documentation.
