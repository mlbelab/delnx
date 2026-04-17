🌳 delnx
=========

.. image:: https://img.shields.io/pypi/v/delnx.svg?color=blue
   :target: https://pypi.org/project/delnx
   :alt: PyPI version

.. image:: https://github.com/joschif/delnx/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/joschif/delnx/actions/workflows/test.yaml
   :alt: Tests

.. image:: https://codecov.io/gh/joschif/delnx/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/joschif/delnx
   :alt: Codecov

.. image:: https://results.pre-commit.ci/badge/github/joschif/delnx/main.svg
   :target: https://results.pre-commit.ci/latest/github/joschif/delnx/main
   :alt: pre-commit.ci status

.. image:: https://img.shields.io/readthedocs/delnx
   :target: https://delnx.readthedocs.io
   :alt: Documentation Status


:mod:`delnx` (``/dɪˈlɒnɪks/ | "de-lo-nix"``) is a python package for differential expression analysis of (single-cell) genomics data. It enables scalable analyses of atlas-level datasets through GPU-accelerated regression models and statistical tests implemented in `JAX <https://docs.jax.dev/en/latest/>`_.

🚀 Installation
---------------

PyPI
~~~~

.. code-block:: bash

   pip install delnx

Development version
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install git+https://github.com/joschif/delnx.git@main

⚡ Quickstart
----------------

Negative binomial DE (count data):

.. code-block:: python

   import delnx as dx

   # Fit negative binomial GLMs with quasi-likelihood dispersion shrinkage
   fit = dx.tl.nb_fit(adata, condition_key="treatment", reference="control")

   # Test for differential expression
   results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]")

General-purpose DE (log-normalized / binary data):

.. code-block:: python

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

💎 Features
------------

- **Negative binomial GLMs**: GPU-accelerated glmGamPoi-style fitting with quasi-likelihood dispersion shrinkage for count data.
- **General-purpose DE**: Logistic regression, ANOVA, and binomial GLM for log-normalized, scaled, or binary data.
- **Formula interface**: R-style formulas (``~ treatment + batch``) parsed by patsy, with treatment coding and reference levels.
- **Rank-based markers**: Fast AUROC-based one-vs-all marker detection with Numba-optimized ranking.
- **Pseudobulking**: Perform DE on large multi-sample datasets by using pseudobulk aggregation.
- **Effect sizes**: Log2 fold change and AUROC computation for pairwise condition comparisons.
- **GPU acceleration**: Core methods are implemented in JAX, enabling GPU acceleration for scalable DE analysis on large datasets.


.. toctree::
    :maxdepth: 3
    :hidden:

    installation
    api
    contributing
    notebooks/index
