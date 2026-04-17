<img src="docs/_static/images/delnx.png" width="300" alt="delnx">


[![PyPI version][badge-pypi]][pypi]
[![Tests][badge-tests]][tests]
[![Codecov][badge-coverage]][codecov]
[![pre-commit.ci status][badge-pre-commit]][pre-commit.ci]
[![Documentation Status][badge-docs]][documentation]


[badge-tests]: https://github.com/joschif/delnx/actions/workflows/test.yaml/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/delnx
[badge-coverage]: https://codecov.io/gh/joschif/delnx/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/joschif/delnx/main.svg
[badge-pypi]: https://img.shields.io/pypi/v/delnx.svg?color=blue


# 🌳 delnx

**delnx** (`"de-lo-nix"  | /dɪˈlɒnɪks/`) is a python package for differential expression analysis of (single-cell) genomics data. It enables scalable analyses of atlas-level datasets through GPU-accelerated regression models and statistical tests implemented in [JAX](https://docs.jax.dev/en/latest/).

## 🚀 Installation

### PyPI

```
pip install delnx
```

### Development version

```bash
pip install git+https://github.com/joschif/delnx.git@main
```


## ⚡ Quickstart

### Negative binomial DE (count data)

```python
import delnx as dx

# Fit negative binomial GLMs with quasi-likelihood dispersion shrinkage
fit = dx.tl.nb_fit(adata, condition_key="treatment", reference="control")

# Test for differential expression
results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]")
```

### General-purpose DE (log-normalized / binary data)

```python
# Logistic regression with likelihood ratio test
results = dx.tl.de(
    adata,
    condition_key="treatment",
    reference="control",
    contrast="treatment[T.drugA]",
)

# Formula-based design with covariates
results = dx.tl.de(
    adata,
    formula="~ treatment + batch",
    contrast="treatment[T.drugA]",
    method="anova",
)
```

### Per-group DE with grouped wrapper

```python
results = dx.tl.grouped(
    dx.tl.de, adata,
    group_key="cell_type",
    condition_key="treatment",
    reference="control",
    contrast="treatment[T.drugA]",
)
```

### Fast rank-based markers

```python
results = dx.tl.rank_de(adata, condition_key="cell_type")
```

## 💎 Features
- **Negative binomial GLMs**: GPU-accelerated glmGamPoi-style fitting with quasi-likelihood dispersion shrinkage for count data.
- **General-purpose DE**: Logistic regression, ANOVA, and binomial GLM for log-normalized, scaled, or binary data.
- **Formula interface**: R-style formulas (`~ treatment + batch`) parsed by patsy, with treatment coding and reference levels.
- **Rank-based markers**: Fast AUROC-based one-vs-all marker detection with Numba-optimized ranking.
- **Pseudobulking**: Perform DE on large multi-sample datasets by using pseudobulk aggregation.
- **Effect sizes**: Log2 fold change and AUROC computation for pairwise condition comparisons.
- **GPU acceleration**: Core methods are implemented in JAX, enabling GPU acceleration for scalable DE analysis on large datasets.


## 📖 Documentation

For more information, check out the [documentation][documentation] and the [API reference][api documentation].



[issue tracker]: https://github.com/joschif/delnx/issues
[tests]: https://github.com/joschif/delnx/actions/workflows/test.yaml
[documentation]: https://delnx.readthedocs.io
[changelog]: https://delnx.readthedocs.io/en/latest/changelog.html
[api documentation]: https://delnx.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/delnx
[codecov]: https://codecov.io/gh/joschif/delnx
[pre-commit.ci]: https://results.pre-commit.ci/latest/github/joschif/delnx/main
