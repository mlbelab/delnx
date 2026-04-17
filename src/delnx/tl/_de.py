"""Differential expression testing for non-count data.

This module provides DE analysis for log-normalized, scaled, or binary data
using logistic regression, ANOVA, or binomial models.

For count data, use :func:`delnx.tl.nb_fit` + :func:`delnx.tl.nb_test`.
For fast cluster markers, use :func:`delnx.tl.rank_de`.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData

from delnx._logging import logger
from delnx._utils import _get_layer

from ._de_tests import _run_de
from ._design import build_design
from ._jax_tests import _run_batched_de
from ._utils import _check_method_and_data_type, _infer_data_type


def de(
    adata: AnnData,
    condition_key: str | None = None,
    formula: str | None = None,
    contrast: str | int | None = None,
    reference: str | None = None,
    covariate_keys: list[str] | None = None,
    method: str = "lr",
    layer: str | None = None,
    multitest_method: str = "fdr_bh",
    batch_size: int = 2048,
    maxiter: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Differential expression testing for non-count data.

    General-purpose DE function for log-normalized, scaled, or binary data.
    Builds a design matrix from ``formula`` or ``condition_key`` and tests
    a single ``contrast``. For multiple comparisons, call this function
    once per contrast or use :func:`grouped` for per-group analysis.

    For count data, use :func:`nb_fit` + :func:`nb_test` instead.
    For fast cluster markers, use :func:`rank_de`.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression data and metadata.
    condition_key : str | None, default=None
        Column name in ``adata.obs`` containing condition labels.
        Internally builds ``"~ condition_key"`` for the design matrix.
        Mutually exclusive with ``formula``.
    formula : str | None, default=None
        R-style formula for the design matrix (e.g., ``"~ treatment + batch"``).
        Parsed by patsy. Mutually exclusive with ``condition_key``.
    contrast : str | int | None, default=None
        Coefficient to test. Can be a design column name
        (e.g., ``"treatment[T.drugA]"``) or an integer index.
        If None, the last coefficient is tested.
    reference : str | None, default=None
        Reference level for categorical conditions. This level becomes
        the intercept in treatment coding.
    covariate_keys : list[str] | None, default=None
        Columns in ``adata.obs`` to include as covariates.
        Only used with ``condition_key`` (include covariates in
        ``formula`` directly).
    method : str, default="lr"
        Statistical method for testing:

            - ``"lr"``: Logistic regression with likelihood ratio test.
              Recommended for log-normalized single-cell data.
            - ``"anova"``: ANOVA based on linear model.
              Recommended for log-normalized or scaled data.
            - ``"anova_residual"``: Linear model with residual F-test.
            - ``"binomial"``: Binomial GLM likelihood ratio test.
              For binary data (e.g., ATAC-seq).
    layer : str | None, default=None
        Layer in ``adata.layers`` containing expression data.
        If None, uses ``adata.X``.
    multitest_method : str, default="fdr_bh"
        Method for multiple testing correction
        (see :func:`statsmodels.stats.multipletests`).
    batch_size : int, default=2048
        Number of features to process per batch.
    maxiter : int, default=100
        Maximum number of optimization iterations.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``feature``: Gene/feature names
        - ``log2fc``: Log2 fold change (coefficient / log(2), clipped to [-10, 10])
        - ``coef``: Model coefficient (log scale)
        - ``stat``: Test statistic (LR chi-squared or F-statistic)
        - ``pval``: Raw p-value
        - ``padj``: Adjusted p-value

    Examples
    --------
    Simple condition comparison:

    >>> results = dx.tl.de(adata, condition_key="treatment", reference="control",
    ...                    contrast="treatment[T.drugA]")

    Formula-based with covariates:

    >>> results = dx.tl.de(adata, formula="~ treatment + batch",
    ...                    contrast="treatment[T.drugA]")

    Continuous covariate:

    >>> results = dx.tl.de(adata, formula="~ age + sex", contrast="age")

    Binomial for binary ATAC data:

    >>> results = dx.tl.de(adata, condition_key="treatment", reference="control",
    ...                    contrast="treatment[T.drugA]", method="binomial",
    ...                    layer="binary")

    Notes
    -----
    For count data (RNA-seq), use :func:`nb_fit` + :func:`nb_test` instead.
    For fast cluster markers, use :func:`rank_de`.
    """
    # Validate mutual exclusivity
    if formula is not None and condition_key is not None:
        raise ValueError("Specify either 'formula' or 'condition_key', not both.")
    if formula is None and condition_key is None:
        raise ValueError("One of 'formula' or 'condition_key' must be specified.")

    # Validate supported methods
    supported_methods = ("lr", "anova", "anova_residual", "binomial")
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported method '{method}'. Use one of {supported_methods}. "
            f"For count data, use nb_fit() + nb_test() instead."
        )

    # Get expression matrix
    X = _get_layer(adata, layer)

    # Infer data type
    data_type = _infer_data_type(X)
    logger.info(f"Inferred data type: {data_type}", verbose=verbose)

    # Validate method and data type combinations
    _check_method_and_data_type(method, data_type)

    # Build design matrix (both paths go through build_design)
    design_matrix, column_names = build_design(
        adata.obs,
        formula=formula,
        condition_key=condition_key,
        reference=reference,
        covariate_keys=covariate_keys,
    )

    # Resolve contrast to column index
    if contrast is None:
        test_idx = design_matrix.shape[1] - 1  # Last coefficient
    elif isinstance(contrast, str):
        if contrast not in column_names:
            raise ValueError(
                f"Contrast '{contrast}' not found in design columns. "
                f"Available: {column_names}"
            )
        test_idx = column_names.index(contrast)
    elif isinstance(contrast, int):
        test_idx = contrast
    else:
        raise NotImplementedError("Custom contrast vectors not yet supported")

    # Filter to non-zero features
    feature_mask = np.array(X.sum(axis=0) > 0).flatten()
    feature_names = adata.var_names[feature_mask].values
    X_filtered = X[:, feature_mask]

    logger.info(f"Running DE for {np.sum(feature_mask)} features", verbose=verbose)

    # Dispatch to backend
    if method == "binomial":
        results = _run_de(
            X=X_filtered,
            model_data=pd.DataFrame(),  # unused with design_matrix
            feature_names=feature_names,
            method=method,
            backend="statsmodels",
            condition_key="",  # unused with design_matrix
            design_matrix=design_matrix,
            test_idx=test_idx,
            verbose=verbose,
        )
    else:
        results = _run_batched_de(
            X=X_filtered,
            model_data=pd.DataFrame(),  # unused with design_matrix
            feature_names=feature_names,
            method=method,
            condition_key="",  # unused with design_matrix
            design_matrix=design_matrix,
            test_idx=test_idx,
            batch_size=batch_size,
            maxiter=maxiter,
            verbose=verbose,
        )

    results["feature"] = results["feature"].astype(str)

    # Check results
    if len(results) == 0 or results["pval"].isna().all():
        raise ValueError(
            "Differential expression analysis failed. "
            "Check input data or set verbose=True for details."
        )

    # Compute log2fc from coefficient (coef is on log scale for LR/binomial)
    results["log2fc"] = np.clip(results["coef"] / np.log(2), -10.0, 10.0)

    # Clip p-values
    results["pval"] = np.clip(results["pval"], 1e-50, 1)

    # Multiple testing correction
    valid = results["pval"].notna()
    results["padj"] = np.nan
    if valid.sum() > 0:
        results.loc[valid, "padj"] = sm.stats.multipletests(
            results.loc[valid, "pval"].values, method=multitest_method
        )[1]

    results = results.sort_values(by=["padj", "coef"]).reset_index(drop=True)
    return results[["feature", "log2fc", "coef", "stat", "pval", "padj"]].copy()
