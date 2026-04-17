"""glmGamPoi-style differential expression analysis.

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
class GLMGPResult:
    """Result container for glm_gp fit.

    Attributes
    ----------
    Beta : np.ndarray
        Fitted coefficients, shape (n_genes, n_coefficients).
    overdispersions : np.ndarray
        MLE dispersion estimates per gene, shape (n_genes,).
    Mu : np.ndarray
        Fitted mean values, shape (n_samples, n_genes).
    size_factors : np.ndarray
        Size factors per sample, shape (n_samples,).
    deviances : np.ndarray
        Deviance per gene, shape (n_genes,).
    design_matrix : np.ndarray
        Design matrix used, shape (n_samples, n_coefficients).
    feature_names : pd.Index
        Feature/gene names.
    counts : np.ndarray | None
        Original count matrix, shape (n_samples, n_genes). Stored for
        reduced-model refitting in test_de.
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

    Beta: np.ndarray
    overdispersions: np.ndarray
    Mu: np.ndarray
    size_factors: np.ndarray
    deviances: np.ndarray
    design_matrix: np.ndarray
    feature_names: pd.Index
    counts: np.ndarray | None = None
    condition_key: str | None = None
    ql_dispersions: np.ndarray | None = None
    df0_prior: float = 0.0
    dispersion_trend: np.ndarray | None = None
    converged: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Main fitting function
# =============================================================================


def glm_gp(
    adata: AnnData,
    condition_key: str | None = None,
    design: np.ndarray | None = None,
    size_factors: str | np.ndarray | None = "normed_sum",
    layer: str | None = None,
    overdispersion: bool = True,
    overdispersion_shrinkage: bool = True,
    do_cox_reid_adjustment: bool = True,
    batch_size: int = 512,
    maxiter: int = 100,
    verbose: bool = True,
) -> GLMGPResult:
    """Fit Gamma-Poisson (negative binomial) GLMs to count data.

    This implements the glmGamPoi approach for fast and accurate differential
    expression analysis using GPU-accelerated Fisher-scoring with quasi-likelihood
    dispersion shrinkage.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing count data.
    condition_key : str | None, default=None
        Column in `adata.obs` for condition labels. Creates a design matrix
        with intercept + condition indicators.
    design : np.ndarray | None, default=None
        Custom design matrix. If provided, overrides condition_key.
        Shape should be (n_samples, n_coefficients).
    size_factors : str | np.ndarray | None, default="normed_sum"
        Size factors for normalization. Can be:
        - "normed_sum": Compute using normalized sum method
        - "poscounts": DESeq2-style positive counts method
        - np.ndarray: Pre-computed size factors
        - None: No size factor normalization (all 1s)
    layer : str | None, default=None
        Layer in `adata.layers` containing counts. If None, uses `adata.X`.
    overdispersion : bool, default=True
        Whether to estimate overdispersion. If False, uses Poisson (disp=0).
    overdispersion_shrinkage : bool, default=True
        Whether to apply quasi-likelihood shrinkage to dispersions.
    do_cox_reid_adjustment : bool, default=True
        Whether to apply Cox-Reid adjustment to dispersion MLE.
    batch_size : int, default=512
        Number of genes to process in each batch.
    maxiter : int, default=100
        Maximum iterations for Fisher-scoring.
    verbose : bool, default=True
        Whether to show progress.

    Returns
    -------
    GLMGPResult
        Fitted model results containing coefficients, dispersions, and
        fitted values.

    Examples
    --------
    Basic usage with condition comparison:

    >>> import delnx as dx
    >>> fit = dx.tl.glm_gp(adata, condition_key="treatment")
    >>> results = dx.tl.test_de(fit, contrast="treatment")
    """
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

    # Build design matrix
    if design is not None:
        design_matrix = np.asarray(design)
    elif condition_key is not None:
        # Create design matrix from condition
        conditions = adata.obs[condition_key].values
        unique_conditions = np.unique(conditions)

        # Intercept + indicators for each condition except reference
        design_matrix = np.ones((n_samples, 1))
        for cond in unique_conditions[1:]:
            col = (conditions == cond).astype(float)
            design_matrix = np.column_stack([design_matrix, col])
    else:
        # Intercept only
        design_matrix = np.ones((n_samples, 1))

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
    Beta = np.zeros((n_genes, n_coef))
    dispersions = np.zeros(n_genes)
    deviances = np.zeros(n_genes)
    converged = np.zeros(n_genes, dtype=bool)
    Mu = np.zeros((n_samples, n_genes))

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
            Beta[start:end, 0] = np.asarray(beta0_arr)
        else:
            beta_arr, dev_arr, conv_arr = fit_beta_newton_batch(
                X_batch, design_jax, offset_jax, disp_arr, beta_arr,
                maxiter, 1e-8,
            )
            eta_batch = design_jax @ beta_arr.T + offset_jax[:, None]
            mu_batch = jnp.exp(jnp.clip(eta_batch, -50, 50))
            Beta[start:end] = np.asarray(beta_arr)

        # Store results
        dispersions[start:end] = np.asarray(disp_arr)
        deviances[start:end] = np.asarray(dev_arr)
        converged[start:end] = np.asarray(conv_arr)
        Mu[:, start:end] = np.asarray(mu_batch)

    # Apply quasi-likelihood shrinkage if requested
    ql_dispersions = None
    df0_prior = 0.0
    dispersion_trend = None

    if overdispersion_shrinkage and overdispersion:
        if verbose:
            logger.info("Applying quasi-likelihood shrinkage", verbose=True)

        # Compute mean expression
        mean_expression = np.mean(Mu, axis=0)

        # Fit dispersion trend
        dispersion_trend = fit_dispersion_trend(
            dispersions, mean_expression, method="local_median"
        )

        # Recompute deviances using trend dispersion (vectorized)
        counts_dense_jax = jnp.array(_to_dense(X).astype(np.float64))
        Mu_jax = jnp.array(Mu)
        disp_trend_jax = jnp.array(dispersion_trend)
        deviances = np.asarray(compute_gp_deviance_batch(
            counts_dense_jax, Mu_jax, disp_trend_jax
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

    # Store dense count matrix for reduced-model refitting in test_de
    counts_dense = _to_dense(X)

    return GLMGPResult(
        Beta=Beta,
        overdispersions=dispersions,
        Mu=Mu,
        size_factors=sf,
        deviances=deviances,
        design_matrix=design_matrix,
        feature_names=adata.var_names,
        counts=counts_dense,
        condition_key=condition_key,
        ql_dispersions=ql_dispersions,
        df0_prior=df0_prior,
        dispersion_trend=dispersion_trend,
        converged=converged,
    )


# =============================================================================
# Differential expression testing
# =============================================================================


def test_de(
    fit: GLMGPResult,
    contrast: str | int | np.ndarray | None = None,
    reduced_design: np.ndarray | None = None,
    pval_adjust_method: str = "fdr_bh",
    lfc_threshold: float = 0.0,
) -> pd.DataFrame:
    """Test for differential expression using quasi-likelihood F-test.

    Parameters
    ----------
    fit : GLMGPResult
        Fitted model from `glm_gp()`.
    contrast : str | int | np.ndarray | None, default=None
        Contrast to test. Can be:
        - str: Name of condition level to test (vs reference)
        - int: Index of coefficient to test
        - np.ndarray: Custom contrast vector
        - None: Test last coefficient
    reduced_design : np.ndarray | None, default=None
        Reduced design matrix for likelihood ratio test.
        If None, automatically created by dropping the tested coefficient.
    pval_adjust_method : str, default="fdr_bh"
        Method for multiple testing correction.
    lfc_threshold : float, default=0.0
        Threshold for log2 fold change filtering.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - feature: Gene/feature names
        - log2fc: Log2 fold change
        - coef: Model coefficient
        - stat: F-statistic
        - pval: Raw p-value
        - padj: Adjusted p-value

    Examples
    --------
    Test for treatment effect:

    >>> fit = dx.tl.glm_gp(adata, condition_key="treatment")
    >>> results = dx.tl.test_de(fit)

    Test specific contrast:

    >>> results = dx.tl.test_de(fit, contrast=1)  # Test second coefficient
    """
    n_genes = fit.Beta.shape[0]
    n_coef = fit.Beta.shape[1]
    n_samples = fit.design_matrix.shape[0]

    # Determine which coefficient to test
    if contrast is None:
        test_idx = n_coef - 1  # Last coefficient
    elif isinstance(contrast, int):
        test_idx = contrast
    elif isinstance(contrast, str):
        # Find coefficient index from condition name
        # This assumes condition_key was used
        if fit.condition_key is None:
            raise ValueError("Cannot use string contrast without condition_key")
        test_idx = n_coef - 1  # Default to last
    else:
        # Custom contrast vector - not implemented yet
        raise NotImplementedError("Custom contrast vectors not yet supported")

    # Get coefficients for tested term
    coefs = fit.Beta[:, test_idx]

    # Log2 fold change (coefficient is on log scale), capped at ±10
    log2fc = np.clip(coefs / np.log(2), -10.0, 10.0)

    # Build reduced design by dropping tested column
    if reduced_design is None:
        keep_cols = [i for i in range(n_coef) if i != test_idx]
        reduced_design = fit.design_matrix[:, keep_cols]

    # Likelihood ratio test: refit reduced model and compare deviances.
    # This matches R's glmGamPoi::test_de approach.
    reduced_jax = jnp.array(reduced_design)
    offset_jax = jnp.array(np.log(np.maximum(fit.size_factors, 1e-10)))
    n_coef_reduced = reduced_design.shape[1]

    if fit.counts is None:
        raise ValueError(
            "Count matrix not stored in GLMGPResult. "
            "Re-run glm_gp to obtain a fit with stored counts."
        )

    # Use dispersion trend if available (matching R), else MLE
    disp_vec = (
        jnp.array(fit.dispersion_trend)
        if fit.dispersion_trend is not None
        else jnp.array(fit.overdispersions)
    )

    counts_jax = jnp.array(fit.counts.astype(np.float64))
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
        init_betas_r = init_betas_r.at[:, 0].set(jnp.array(fit.Beta[:, 0]))
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
            pvals[valid_pvals], method=pval_adjust_method
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
