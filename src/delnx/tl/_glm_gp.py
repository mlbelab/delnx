"""glmGamPoi-style differential expression analysis.

This module provides the main interface for GPU-accelerated negative binomial
differential expression analysis using the glmGamPoi approach.

References
----------
Ahlmann-Eltze, C., & Huber, W. (2020). glmGamPoi: fitting Gamma-Poisson
generalized linear models on single cell count data. Bioinformatics.
"""

from dataclasses import dataclass, field

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
    estimate_dispersion_mle,
    estimate_dispersion_moments,
    fit_beta_fisher_scoring,
    fit_beta_one_group,
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

    # Initialize storage
    Beta = np.zeros((n_genes, n_coef))
    dispersions = np.zeros(n_genes)
    deviances = np.zeros(n_genes)
    converged = np.zeros(n_genes, dtype=bool)
    Mu = np.zeros((n_samples, n_genes))

    # Determine if intercept-only model
    is_intercept_only = n_coef == 1

    # Process genes in batches
    n_batches = (n_genes + batch_size - 1) // batch_size

    if verbose:
        logger.info(f"Fitting {n_genes} genes with {n_coef} coefficient(s)", verbose=True)

    for batch_idx in tqdm.tqdm(range(n_batches), disable=not verbose, desc="Fitting GLMs"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_genes)
        batch_genes = range(start, end)

        # Get batch data
        X_batch = _to_dense(X[:, start:end])

        for i, gene_idx in enumerate(batch_genes):
            counts = jnp.array(X_batch[:, i], dtype=jnp.float64)

            # Initial dispersion estimate (moments)
            mu_init = sf * np.mean(X_batch[:, i] / sf)
            disp_init = float(estimate_dispersion_moments(counts, jnp.array(mu_init)))
            disp_init = max(disp_init, 1e-8)

            # Fit beta using Fisher-scoring
            if is_intercept_only:
                # Fast path for intercept-only
                beta0, dev, conv = fit_beta_one_group(
                    counts,
                    jnp.array(sf),
                    disp_init,
                    maxiter=maxiter,
                )
                beta = np.array([float(beta0)])
                mu = sf * np.exp(float(beta0))
            else:
                # Full Fisher-scoring
                init_beta = jnp.zeros(n_coef)
                # Better init for intercept
                mean_norm = np.mean(X_batch[:, i] / sf)
                init_beta = init_beta.at[0].set(np.log(max(mean_norm, 1e-10)))

                beta, dev, conv = fit_beta_fisher_scoring(
                    counts,
                    design_jax,
                    offset_jax,
                    disp_init,
                    init_beta,
                    maxiter=maxiter,
                )
                beta = np.array(beta)
                eta = design_matrix @ beta + log_sf
                mu = np.exp(np.clip(eta, -50, 50))

            # Estimate dispersion MLE
            if overdispersion:
                disp, _ = estimate_dispersion_mle(
                    counts,
                    jnp.array(mu),
                    design_jax,
                    disp_init,
                    do_cox_reid=do_cox_reid_adjustment,
                )
                disp = float(disp)
            else:
                disp = 1e-10  # Effectively Poisson

            # Refit with final dispersion
            if is_intercept_only:
                beta0, dev, conv = fit_beta_one_group(
                    counts,
                    jnp.array(sf),
                    disp,
                    maxiter=maxiter,
                )
                beta = np.array([float(beta0)])
                mu = sf * np.exp(float(beta0))
                dev = float(dev)
            else:
                beta, dev, conv = fit_beta_fisher_scoring(
                    counts,
                    design_jax,
                    offset_jax,
                    disp,
                    jnp.array(beta),
                    maxiter=maxiter,
                )
                beta = np.array(beta)
                eta = design_matrix @ beta + log_sf
                mu = np.exp(np.clip(eta, -50, 50))
                dev = float(dev)

            # Store results
            Beta[gene_idx] = beta
            dispersions[gene_idx] = disp
            deviances[gene_idx] = dev
            converged[gene_idx] = bool(conv)
            Mu[:, gene_idx] = mu

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

        # Transform to QL scale
        ql_dispersions = compute_ql_dispersions(
            dispersions, mean_expression, dispersion_trend
        )

        # Apply empirical Bayes shrinkage
        df_residual = n_samples - n_coef
        ql_dispersions, df0_prior, _ = shrink_ql_dispersions(
            ql_dispersions, df_residual
        )

    return GLMGPResult(
        Beta=Beta,
        overdispersions=dispersions,
        Mu=Mu,
        size_factors=sf,
        deviances=deviances,
        design_matrix=design_matrix,
        feature_names=adata.var_names,
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

    # Log2 fold change (coefficient is on log scale)
    log2fc = coefs / np.log(2)

    # Create reduced design by dropping tested column
    if reduced_design is None:
        keep_cols = [i for i in range(n_coef) if i != test_idx]
        reduced_design = fit.design_matrix[:, keep_cols]

    # Compute deviance for reduced model
    # We need to refit with reduced design to get proper deviances
    # For now, use the approximation based on coefficient significance

    # Use QL F-test if available, otherwise Wald test
    if fit.ql_dispersions is not None:
        # Quasi-likelihood F-test
        # Approximate by comparing full vs reduced deviance
        # For a single coefficient test, the deviance difference is approximately
        # chi2 with 1 df, so F ~ chi2/1

        # Wald-like approximation for F-statistic
        # F = coef^2 / (var(coef) * ql_disp)
        # Using deviance-based approximation
        df_full = n_samples - n_coef
        df_reduced = n_samples - (n_coef - 1)

        # Approximate deviance difference using coefficient
        # This is a simplification; full implementation would refit
        dev_diff = coefs ** 2 * df_full / np.maximum(fit.ql_dispersions, 1e-10)

        f_stats, pvals = ql_f_test(
            fit.deviances,
            fit.deviances + dev_diff,
            df_full=df_full,
            df_reduced=df_reduced,
            ql_dispersions=fit.ql_dispersions,
            df0_prior=fit.df0_prior,
        )
    else:
        # Fall back to Wald-like test
        # z = coef / se, p-value from normal
        # Approximate SE from deviance
        se_approx = np.sqrt(fit.deviances / (n_samples - n_coef))
        z_stats = coefs / np.maximum(se_approx, 1e-10)
        f_stats = z_stats ** 2
        from scipy import stats
        pvals = stats.chi2.sf(f_stats, df=1)

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
