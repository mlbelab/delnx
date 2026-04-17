"""Negative binomial differential expression analysis.

This module provides the main interface for GPU-accelerated negative binomial
differential expression analysis using the glmGamPoi approach. Core solvers
are vectorized with JAX vmap for batch-parallel processing across genes.

References
----------
Ahlmann-Eltze, C., & Huber, W. (2020). glmGamPoi: fitting Gamma-Poisson
generalized linear models on single cell count data. Bioinformatics.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
from anndata import AnnData
from scipy import sparse

from delnx._logging import logger
from delnx._utils import _get_layer, _to_dense
from delnx.models._glm_gp import (
    compute_gp_deviance,
    compute_gp_deviance_batch,
    estimate_dispersion_mle,
    estimate_dispersion_mle_batch,
    estimate_dispersion_moments,
    estimate_dispersion_moments_batch,
    fit_beta_fisher_scoring,
    fit_beta_newton_batch,
    fit_beta_one_group,
    fit_beta_one_group_batch,
)
from delnx.models._quasi_likelihood import (
    compute_ql_dispersions,
    fit_dispersion_trend,
    ql_f_test,
    shrink_ql_dispersions,
)
from delnx.pp._size_factors import size_factors as compute_size_factors


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class NBFitResult:
    """Result container for nb_fit.

    Attributes
    ----------
    beta : np.ndarray
        Fitted coefficients, shape (n_genes, n_coefficients).
    overdispersions : np.ndarray
        MLE dispersion estimates per gene, shape (n_genes,).
    mu : np.ndarray
        Fitted mean values, shape (n_samples, n_genes).
    size_factors : np.ndarray
        Size factors per sample, shape (n_samples,).
    deviances : np.ndarray
        Deviance per gene, shape (n_genes,).
    design_matrix : np.ndarray
        Design matrix used, shape (n_samples, n_coefficients).
    design_column_names : list[str]
        Column names for the design matrix (e.g., ["intercept", "condB"]).
    feature_names : pd.Index
        Feature/gene names.
    layer : str | None
        Layer used for count data.
    condition_key : str | None
        Condition key used for design.
    ql_dispersions : np.ndarray | None
        Quasi-likelihood dispersions (if shrinkage applied).
    df0_prior : float
        Prior degrees of freedom from QL shrinkage.
    dispersion_trend : np.ndarray | None
        Fitted dispersion trend.
    converged : np.ndarray
        Convergence status per gene.
    """

    beta: np.ndarray
    overdispersions: np.ndarray
    mu: np.ndarray
    size_factors: np.ndarray
    deviances: np.ndarray
    design_matrix: np.ndarray
    design_column_names: list[str]
    feature_names: pd.Index
    layer: str | None = None
    condition_key: str | None = None
    ql_dispersions: np.ndarray | None = None
    df0_prior: float = 0.0
    dispersion_trend: np.ndarray | None = None
    converged: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Main fitting function
# =============================================================================


def nb_fit(
    adata: AnnData,
    condition_key: str | None = None,
    formula: str | None = None,
    design: np.ndarray | None = None,
    design_column_names: list[str] | None = None,
    reference: str | None = None,
    covariate_keys: list[str] | None = None,
    size_factors: str | np.ndarray | None = "normed_sum",
    layer: str | None = None,
    overdispersion: bool = True,
    batch_size: int = 512,
    maxiter: int = 100,
    verbose: bool = True,
    overdispersion_shrinkage: bool = True,
    do_cox_reid_adjustment: bool = True,
) -> NBFitResult:
    """Fit Gamma-Poisson (negative binomial) GLMs to count data.

    This implements the glmGamPoi approach for fast and accurate differential
    expression analysis using GPU-accelerated Newton-Raphson with quasi-likelihood
    dispersion shrinkage.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing count data.
    condition_key : str | None, default=None
        Column in ``adata.obs`` for condition labels. Creates a design matrix
        with intercept + condition indicators. Mutually exclusive with ``formula``.
    formula : str | None, default=None
        R-style formula for the design matrix (e.g., ``"~ treatment + batch"``
        or ``"~ treatment * batch"``). Parsed by patsy. Mutually exclusive
        with ``condition_key``.
    design : np.ndarray | None, default=None
        Custom design matrix. If provided, overrides ``condition_key``,
        ``formula``, ``reference``, and ``covariate_keys``.
        Shape should be (n_samples, n_coefficients).
    design_column_names : list[str] | None, default=None
        Column names for a custom ``design`` matrix. Enables string-based
        ``contrast`` in :func:`nb_test`. If None with custom ``design``,
        generic names ``coef_0``, ``coef_1``, ... are used.
    reference : str | None, default=None
        Reference level for the condition. This level becomes the intercept.
        If None, the alphabetically first level is used.
    covariate_keys : list[str] | None, default=None
        Columns in ``adata.obs`` to include as covariates in the design matrix.
        Only used with ``condition_key`` (include covariates in ``formula`` directly).
    size_factors : str | np.ndarray | None, default="normed_sum"
        Size factors for normalization. Can be:

        - ``"normed_sum"``: Compute using normalized sum method
        - ``"poscounts"``: DESeq2-style positive counts method
        - np.ndarray: Pre-computed size factors
        - None: No size factor normalization (all 1s)
    layer : str | None, default=None
        Layer in ``adata.layers`` containing counts. If None, uses ``adata.X``.
    overdispersion : bool, default=True
        Whether to estimate overdispersion. If False, uses Poisson (disp=0).
    batch_size : int, default=512
        Number of genes to process in each batch.
    maxiter : int, default=100
        Maximum iterations for Newton-Raphson.
    verbose : bool, default=True
        Whether to show progress.
    overdispersion_shrinkage : bool, default=True
        Whether to apply quasi-likelihood shrinkage to dispersions.
    do_cox_reid_adjustment : bool, default=True
        Whether to apply Cox-Reid adjustment to dispersion MLE.

    Returns
    -------
    NBFitResult
        Fitted model results containing coefficients, dispersions, and
        fitted values.

    Examples
    --------
    Basic usage with condition comparison:

    >>> import delnx as dx
    >>> fit = dx.tl.nb_fit(adata, condition_key="treatment")
    >>> results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]")

    With reference level and covariates:

    >>> fit = dx.tl.nb_fit(adata, condition_key="treatment", reference="control",
    ...                    covariate_keys=["batch", "sex"])

    Formula-based design with interactions:

    >>> fit = dx.tl.nb_fit(adata, formula="~ treatment * batch")
    >>> results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]:batch[T.batch2]")

    Continuous covariate:

    >>> fit = dx.tl.nb_fit(adata, formula="~ age + sex")
    >>> results = dx.tl.nb_test(adata, fit, contrast="age")
    """
    from ._design import build_design

    # Get count matrix
    X = _get_layer(adata, layer)
    n_samples, n_genes = X.shape

    # Compute or validate size factors
    if size_factors is None:
        sf = np.ones(n_samples)
    elif isinstance(size_factors, str):
        # compute_size_factors stores result in adata.obs
        compute_size_factors(adata, layer=layer, method=size_factors)
        sf = adata.obs["size_factors"].values.astype(float)
    else:
        sf = np.asarray(size_factors)

    sf = np.maximum(sf, 1e-10)
    log_sf = np.log(sf)

    # Build design matrix (priority: design > formula > condition_key)
    if design is not None:
        design_matrix = np.asarray(design)
        if design_column_names is None:
            design_column_names = [f"coef_{i}" for i in range(design_matrix.shape[1])]
    elif formula is not None or condition_key is not None:
        design_matrix, design_column_names = build_design(
            adata.obs,
            formula=formula,
            condition_key=condition_key,
            reference=reference,
            covariate_keys=covariate_keys,
        )
    else:
        # Intercept only
        design_matrix = np.ones((n_samples, 1))
        design_column_names = ["Intercept"]

    n_coef = design_matrix.shape[1]

    # Convert to JAX arrays
    design_jax = jnp.array(design_matrix)
    offset_jax = jnp.array(log_sf)

    # Determine if intercept-only model
    is_intercept_only = n_coef == 1

    # Pre-compute QR decomposition for beta initialization
    # (matching R's estimate_betas_roughly: OLS on log-normalized counts)
    if not is_intercept_only:
        Q_init, R_init = np.linalg.qr(design_matrix)
        Q_init = Q_init[:, :n_coef]
        R_init = R_init[:n_coef, :]

    # Process genes in vectorized batches
    n_batches = (n_genes + batch_size - 1) // batch_size

    if verbose:
        logger.info(f"Fitting {n_genes} genes with {n_coef} coefficient(s)", verbose=True)

    # Initialize storage
    all_beta = np.zeros((n_genes, n_coef))
    dispersions = np.zeros(n_genes)
    deviances = np.zeros(n_genes)
    converged = np.zeros(n_genes, dtype=bool)
    all_mu = np.zeros((n_samples, n_genes))

    sf_jax = jnp.array(sf)

    for batch_idx in tqdm.tqdm(range(n_batches), disable=not verbose, desc="Fitting GLMs"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_genes)
        b = end - start  # batch width

        # Get batch data as dense JAX array (n_samples, b)
        X_batch_np = _to_dense(X[:, start:end]).astype(np.float64)
        X_batch = jnp.array(X_batch_np)

        # --- Step 1: Initial beta fit with moment-based dispersion ---
        if is_intercept_only:
            # Moment dispersion: broadcast sf across genes
            mu_init = sf_jax[:, None] * jnp.mean(X_batch / sf_jax[:, None], axis=0, keepdims=True)
            disp_init = estimate_dispersion_moments_batch(X_batch, mu_init)
            disp_init = jnp.maximum(disp_init, 1e-8)

            # Batch fit intercept-only
            beta0_arr, dev_arr, conv_arr = fit_beta_one_group_batch(
                X_batch, sf_jax, disp_init, maxiter, 1e-8,
            )
            # Compute mu from intercept
            mu_batch = sf_jax[:, None] * jnp.exp(beta0_arr[None, :])
        else:
            # Moment dispersion
            mu_init = sf_jax[:, None] * jnp.mean(X_batch / sf_jax[:, None], axis=0, keepdims=True)
            disp_init = estimate_dispersion_moments_batch(X_batch, mu_init)
            disp_init = jnp.maximum(disp_init, 1e-8)

            # OLS initialization for all genes in batch
            log_norm = np.log(X_batch_np / sf[:, None] + 1.0)  # (n_samples, b)
            init_betas = np.linalg.solve(R_init, Q_init.T @ log_norm)  # (n_coef, b)
            init_betas_jax = jnp.array(init_betas.T)  # (b, n_coef)

            # Batch Newton-Raphson fit
            beta_arr, dev_arr, conv_arr = fit_beta_newton_batch(
                X_batch, design_jax, offset_jax, disp_init, init_betas_jax,
                maxiter, 1e-8,
            )
            # Compute mu
            # beta_arr: (b, n_coef), design_jax: (n_samples, n_coef)
            eta_batch = design_jax @ beta_arr.T + offset_jax[:, None]  # (n_samples, b)
            mu_batch = jnp.exp(jnp.clip(eta_batch, -50, 50))

        # --- Step 2: Dispersion MLE ---
        if overdispersion:
            disp_arr, _ = estimate_dispersion_mle_batch(
                X_batch, mu_batch, design_jax, disp_init,
                do_cox_reid_adjustment, 50, 1e-6,
            )
        else:
            disp_arr = jnp.full(b, 1e-10)

        # --- Step 3: Refit beta with final dispersion ---
        if is_intercept_only:
            beta0_arr, dev_arr, conv_arr = fit_beta_one_group_batch(
                X_batch, sf_jax, disp_arr, maxiter, 1e-8,
            )
            mu_batch = sf_jax[:, None] * jnp.exp(beta0_arr[None, :])
            all_beta[start:end, 0] = np.asarray(beta0_arr)
        else:
            beta_arr, dev_arr, conv_arr = fit_beta_newton_batch(
                X_batch, design_jax, offset_jax, disp_arr, beta_arr,
                maxiter, 1e-8,
            )
            eta_batch = design_jax @ beta_arr.T + offset_jax[:, None]
            mu_batch = jnp.exp(jnp.clip(eta_batch, -50, 50))
            all_beta[start:end] = np.asarray(beta_arr)

        # Store results
        dispersions[start:end] = np.asarray(disp_arr)
        deviances[start:end] = np.asarray(dev_arr)
        converged[start:end] = np.asarray(conv_arr)
        all_mu[:, start:end] = np.asarray(mu_batch)

    # Apply quasi-likelihood shrinkage if requested
    ql_dispersions = None
    df0_prior = 0.0
    dispersion_trend = None

    if overdispersion_shrinkage and overdispersion:
        if verbose:
            logger.info("Applying quasi-likelihood shrinkage", verbose=True)

        # Compute mean expression
        mean_expression = np.mean(all_mu, axis=0)

        # Fit dispersion trend
        dispersion_trend = fit_dispersion_trend(
            dispersions, mean_expression, method="local_median"
        )

        # Recompute deviances using trend dispersion (vectorized)
        counts_dense_jax = jnp.array(_to_dense(X).astype(np.float64))
        mu_jax = jnp.array(all_mu)
        disp_trend_jax = jnp.array(dispersion_trend)
        deviances = np.asarray(compute_gp_deviance_batch(
            counts_dense_jax, mu_jax, disp_trend_jax
        ))

        # Transform to QL scale
        ql_dispersions = compute_ql_dispersions(
            dispersions, mean_expression, dispersion_trend
        )

        # Apply empirical Bayes shrinkage
        df_residual = n_samples - n_coef
        ql_dispersions, df0_prior, _ = shrink_ql_dispersions(
            ql_dispersions, df_residual
        )

    return NBFitResult(
        beta=all_beta,
        overdispersions=dispersions,
        mu=all_mu,
        size_factors=sf,
        deviances=deviances,
        design_matrix=design_matrix,
        design_column_names=design_column_names,
        feature_names=adata.var_names,
        layer=layer,
        condition_key=condition_key,
        ql_dispersions=ql_dispersions,
        df0_prior=df0_prior,
        dispersion_trend=dispersion_trend,
        converged=converged,
    )


# =============================================================================
# Differential expression testing
# =============================================================================


def nb_test(
    adata: AnnData,
    fit: NBFitResult,
    contrast: str | int | None = None,
    reduced_design: np.ndarray | None = None,
    multitest_method: str = "fdr_bh",
    lfc_threshold: float = 0.0,
) -> pd.DataFrame:
    """Test for differential expression using quasi-likelihood F-test.

    Parameters
    ----------
    adata : AnnData
        AnnData object (same one passed to ``nb_fit()``).
    fit : NBFitResult
        Fitted model from ``nb_fit()``.
    contrast : str | int | None, default=None
        Contrast to test. Can be:

        - str: Design column name to test (e.g., ``"treatmentB"``).
          Use ``fit.design_column_names`` to see available names.
        - int: Index of coefficient to test.
        - None: Test last coefficient.
    reduced_design : np.ndarray | None, default=None
        Reduced design matrix for likelihood ratio test.
        If None, automatically created by dropping the tested coefficient.
    multitest_method : str, default="fdr_bh"
        Method for multiple testing correction.
    lfc_threshold : float, default=0.0
        Threshold for log2 fold change filtering.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``feature``: Gene/feature names
        - ``log2fc``: Log2 fold change
        - ``coef``: Model coefficient
        - ``stat``: F-statistic
        - ``pval``: Raw p-value
        - ``padj``: Adjusted p-value

    Examples
    --------
    Test for treatment effect:

    >>> fit = dx.tl.nb_fit(adata, condition_key="treatment")
    >>> results = dx.tl.nb_test(adata, fit)

    Test specific contrast by name:

    >>> results = dx.tl.nb_test(adata, fit, contrast="treatmentB")
    """
    n_genes = fit.beta.shape[0]
    n_coef = fit.beta.shape[1]
    n_samples = fit.design_matrix.shape[0]

    # Determine which coefficient to test
    if contrast is None:
        test_idx = n_coef - 1  # Last coefficient
    elif isinstance(contrast, int):
        test_idx = contrast
    elif isinstance(contrast, str):
        if contrast not in fit.design_column_names:
            raise ValueError(
                f"Contrast '{contrast}' not found in design columns. "
                f"Available: {fit.design_column_names}"
            )
        test_idx = fit.design_column_names.index(contrast)
    else:
        raise NotImplementedError("Custom contrast vectors not yet supported")

    # Get coefficients for tested term
    coefs = fit.beta[:, test_idx]

    # Log2 fold change (coefficient is on log scale), capped at ±10
    log2fc = np.clip(coefs / np.log(2), -10.0, 10.0)

    # Build reduced design by dropping tested column
    if reduced_design is None:
        keep_cols = [i for i in range(n_coef) if i != test_idx]
        reduced_design = fit.design_matrix[:, keep_cols]

    # Likelihood ratio test: refit reduced model and compare deviances.
    reduced_jax = jnp.array(reduced_design)
    offset_jax = jnp.array(np.log(np.maximum(fit.size_factors, 1e-10)))
    n_coef_reduced = reduced_design.shape[1]

    # Use dispersion trend if available (matching R), else MLE
    disp_vec = (
        jnp.array(fit.dispersion_trend)
        if fit.dispersion_trend is not None
        else jnp.array(fit.overdispersions)
    )

    # Get counts from adata (not stored on fit)
    X = _get_layer(adata, fit.layer)
    counts_jax = jnp.array(_to_dense(X).astype(np.float64))
    sf_jax = jnp.array(fit.size_factors)

    # Batch refit reduced model
    if n_coef_reduced == 1:
        # Intercept-only reduced model — use vmapped fast path
        _, deviances_reduced_jax, _ = fit_beta_one_group_batch(
            counts_jax, sf_jax, disp_vec, 100, 1e-8,
        )
        deviances_reduced = np.asarray(deviances_reduced_jax)
    else:
        # Multi-coef reduced model — batch Newton-Raphson
        init_betas_r = jnp.zeros((n_genes, n_coef_reduced))
        # Initialize intercept from full model
        init_betas_r = init_betas_r.at[:, 0].set(jnp.array(fit.beta[:, 0]))
        _, deviances_reduced_jax, _ = fit_beta_newton_batch(
            counts_jax, reduced_jax, offset_jax, disp_vec, init_betas_r,
            100, 1e-8,
        )
        deviances_reduced = np.asarray(deviances_reduced_jax)

    # F-statistic from deviance difference
    df_full = n_samples - n_coef
    df_test = n_coef - n_coef_reduced  # typically 1

    dev_diff = np.maximum(deviances_reduced - fit.deviances, 0.0)

    if fit.ql_dispersions is not None:
        # Quasi-likelihood F-test
        f_stats = dev_diff / (df_test * np.maximum(fit.ql_dispersions, 1e-10))
        df_denom = df_full + fit.df0_prior
        from scipy import stats
        pvals = stats.f.sf(f_stats, df_test, df_denom)
    else:
        # Without QL shrinkage, use chi-squared approximation
        f_stats = dev_diff / df_test
        from scipy import stats
        pvals = stats.chi2.sf(f_stats, df=df_test)

    # Multiple testing correction
    valid_pvals = np.isfinite(pvals)
    padj = np.ones_like(pvals)
    if valid_pvals.sum() > 0:
        padj[valid_pvals] = sm.stats.multipletests(
            pvals[valid_pvals], method=multitest_method
        )[1]

    # Create results DataFrame
    results = pd.DataFrame({
        "feature": fit.feature_names,
        "log2fc": log2fc,
        "coef": coefs,
        "stat": f_stats,
        "pval": pvals,
        "padj": padj,
    })

    # Apply LFC threshold if specified
    if lfc_threshold > 0:
        results = results[np.abs(results["log2fc"]) >= lfc_threshold]

    # Sort by p-value
    results = results.sort_values("pval").reset_index(drop=True)

    return results


def nb_de(
    adata: AnnData,
    condition_key: str | None = None,
    formula: str | None = None,
    design: np.ndarray | None = None,
    design_column_names: list[str] | None = None,
    reference: str | None = None,
    covariate_keys: list[str] | None = None,
    size_factors: str | np.ndarray | None = "normed_sum",
    layer: str | None = None,
    contrast: str | int | None = None,
    multitest_method: str = "fdr_bh",
    lfc_threshold: float = 0.0,
    overdispersion: bool = True,
    batch_size: int = 512,
    maxiter: int = 100,
    verbose: bool = True,
    overdispersion_shrinkage: bool = True,
    do_cox_reid_adjustment: bool = True,
) -> pd.DataFrame:
    """One-shot negative binomial DE: fit model and test in one call.

    Convenience wrapper around :func:`nb_fit` + :func:`nb_test`.
    For reusing a fit across multiple contrasts, call them separately.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing count data.
    condition_key : str | None, default=None
        Column in ``adata.obs`` for condition labels. Mutually exclusive
        with ``formula``.
    formula : str | None, default=None
        R-style formula for the design matrix (e.g., ``"~ treatment + batch"``).
        Mutually exclusive with ``condition_key``.
    design : np.ndarray | None, default=None
        Custom design matrix. Overrides ``condition_key`` and ``formula``.
    design_column_names : list[str] | None, default=None
        Column names for a custom ``design`` matrix.
    reference : str | None, default=None
        Reference level for the condition.
    covariate_keys : list[str] | None, default=None
        Columns in ``adata.obs`` to include as covariates.
    size_factors : str | np.ndarray | None, default="normed_sum"
        Size factors for normalization.
    layer : str | None, default=None
        Layer in ``adata.layers`` containing counts.
    contrast : str | int | None, default=None
        Contrast to test (passed to :func:`nb_test`).
    multitest_method : str, default="fdr_bh"
        Method for multiple testing correction.
    lfc_threshold : float, default=0.0
        Minimum absolute log2 fold change threshold.
    overdispersion : bool, default=True
        Whether to estimate overdispersion.
    batch_size : int, default=512
        Number of genes per batch.
    maxiter : int, default=100
        Maximum iterations for Newton-Raphson.
    verbose : bool, default=True
        Whether to show progress.
    overdispersion_shrinkage : bool, default=True
        Whether to apply quasi-likelihood shrinkage.
    do_cox_reid_adjustment : bool, default=True
        Whether to apply Cox-Reid adjustment.

    Returns
    -------
    pd.DataFrame
        DE results (same as :func:`nb_test`).

    Examples
    --------
    Simple condition comparison:

    >>> results = dx.tl.nb_de(adata, condition_key="treatment", reference="control")

    Formula-based with covariates:

    >>> results = dx.tl.nb_de(adata, formula="~ treatment + batch",
    ...                       contrast="treatment[T.drugA]")
    """
    fit = nb_fit(
        adata,
        condition_key=condition_key,
        formula=formula,
        design=design,
        design_column_names=design_column_names,
        reference=reference,
        covariate_keys=covariate_keys,
        size_factors=size_factors,
        layer=layer,
        overdispersion=overdispersion,
        batch_size=batch_size,
        maxiter=maxiter,
        verbose=verbose,
        overdispersion_shrinkage=overdispersion_shrinkage,
        do_cox_reid_adjustment=do_cox_reid_adjustment,
    )
    return nb_test(
        adata,
        fit,
        contrast=contrast,
        multitest_method=multitest_method,
        lfc_threshold=lfc_threshold,
    )


# Deprecated aliases
GLMGPResult = NBFitResult
glm_gp = nb_fit
glm_gp_test = nb_test
