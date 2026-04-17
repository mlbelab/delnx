# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**delnx** is a Python package for differential expression (DE) analysis of single-cell genomics data. It provides GPU-accelerated regression models and statistical tests via JAX, with statsmodels as a fallback for binomial GLMs. Works with AnnData objects from the scverse ecosystem.

## Common Commands

### Testing
```bash
hatch test                    # run tests (highest supported Python)
hatch test --all              # run tests across all supported Python versions (3.11, 3.13)
hatch test -- tests/test_de.py            # run a single test file
hatch test -- tests/test_de.py::test_name # run a single test
```

### Linting & Formatting
```bash
pre-commit run --all-files    # run all pre-commit checks (ruff, biome, pyproject-fmt)
ruff check src/               # lint only
ruff format src/              # format only
```

### Documentation
```bash
hatch run docs:build          # build sphinx docs
hatch run docs:open           # open built docs in browser
```

## Architecture

The package lives in `src/delnx/` and follows a scanpy-style modular API: `import delnx as dx`, then `dx.pp.*`, `dx.tl.*`, `dx.pl.*`, `dx.ds.*`.

### Submodules

- **`pp`** (preprocessing): pseudobulk aggregation, size factor estimation, AUCell scoring
- **`tl`** (tools): `de()` for general-purpose DE (LR, ANOVA, binomial), `nb_fit()`/`nb_test()` for negative binomial GLMs, `rank_de()` for fast AUROC markers, `grouped()` for per-group DE, `log2fc()`/`auroc()` for effect sizes, `build_design()` for formula-based design matrices
- **`pl`** (plotting): volcano, heatmap, dot, matrix, and violin plots (built on marsilea). All plot classes inherit from `_baseplot.BasePlot`
- **`models`**: JAX-based regression models (`NegativeBinomialRegression`, `LogisticRegression`, `LinearRegression`) and glmGamPoi core solvers (`_glm_gp.py`, `_quasi_likelihood.py`)
- **`ds`** (datasets): synthetic data generation, GMT gene set loading

### Key Data Flow

**Count data (RNA-seq):**
1. `fit = dx.tl.nb_fit(adata, condition_key="treatment", reference="control")` - fit NB GLMs
2. `results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]")` - test contrast

**Non-count data (log-normalized, binary):**
1. `results = dx.tl.de(adata, condition_key="treatment", reference="control", contrast="treatment[T.drugA]")` - builds design matrix, tests contrast

Both paths support R-style formulas via `formula="~ treatment + batch"` and share `build_design()` for design matrix construction. `de()` dispatches to JAX (`_jax_tests.py`) for LR/ANOVA or statsmodels (`_de_tests.py`) for binomial.

## Development Environment

The primary dev environment is the conda env `py313_delnx`. To activate:
```bash
ml Miniforge3
eval "$(conda shell.bash hook)"
conda activate py313_delnx
```

This env includes:
- Python 3.13 with all delnx dependencies (installed via `uv pip install -e ".[test]"`)
- R 4.5 with `glmGamPoi` (via BiocManager) and `rpy2` for cross-validation testing
- Use `mamba` (not `conda`) for package management where possible

### Cross-validation against R glmGamPoi
```bash
conda activate py313_delnx
python scripts/test_vs_glmgampoi.py
```

## Code Conventions

- **Ruff** for linting and formatting, line length 120, numpy-style docstrings
- **Biome** for JSON formatting
- Source layout: `src/delnx/` (uses `--import-mode=importlib` for pytest)
- Build system: hatchling; environments managed with hatch (installer: uv)
- Python >=3.11
- Docstring rules: numpy convention, no module/package/`__init__`/`__magic__` docstrings required
