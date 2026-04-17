"""Batched differential expression testing with JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import tqdm
from scipy import sparse

from delnx._utils import _to_dense
from delnx.models import LogisticRegression


@partial(jax.jit, static_argnums=(3, 4))
def _fit_lr(y, covars, x=None, optimizer="BFGS", maxiter=100):
    """Fit a single logistic regression model using JAX."""
    model = LogisticRegression(skip_stats=True, optimizer=optimizer, maxiter=maxiter)

    if x is not None:
        X = jnp.column_stack([covars, x])
    else:
        X = covars

    results = model.fit(X, y)
    return results["llf"], results["coef"]


_fit_lr_batch = jax.vmap(_fit_lr, in_axes=(None, None, 1, None, None), out_axes=(0, 0))


def _run_lr_test(
    X: jnp.ndarray,
    cond: jnp.ndarray,
    covars: jnp.ndarray | None = None,
    optimizer: str = "BFGS",
    maxiter: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run logistic regression LR test for a batch of features."""
    # Fit null model (shared across features)
    ll_null, _ = _fit_lr(cond, covars, optimizer=optimizer, maxiter=maxiter)

    # Vectorized fit of full models
    ll_full, coefs_full = _fit_lr_batch(cond, covars, X, optimizer, maxiter)

    lr_stats = 2 * (ll_full - ll_null)
    pvals = stats.chi2.sf(lr_stats, 1)
    coefs = coefs_full[:, -1]

    return coefs, np.asarray(lr_stats), pvals


# =============================================================================
# Closed-form ANOVA F-test (no iterative fitting)
# =============================================================================


@jax.jit
def _anova_precompute(covars):
    """Precompute null-model projection matrix (shared across all genes).

    Returns the residual-maker M_null = I - H_null where H_null is the
    hat matrix of the covariate (null) model.
    """
    # QR is numerically stable for computing projections
    Q, _ = jnp.linalg.qr(covars)
    # M_null = I - Q @ Q^T (residual maker)
    M_null = jnp.eye(covars.shape[0]) - Q @ Q.T
    return M_null


@jax.jit
def _anova_ftest_single(x, y, M_null, n, p_null):
    """Closed-form ANOVA F-test for a single feature.

    Tests whether adding feature x to the null model (covars only)
    significantly improves fit. Uses the Frisch-Waugh-Lovell theorem:
    the coefficient and SS reduction from adding x equal those from
    regressing M_null @ y on M_null @ x.

    Parameters
    ----------
    x : feature values (n_samples,)
    y : response (n_samples,)
    M_null : null-model residual maker (n_samples, n_samples)
    n : number of samples
    p_null : number of null-model columns
    """
    # Project out null-model covariates
    x_resid = M_null @ x      # (n,)
    y_resid = M_null @ y      # (n,)

    # OLS of y_resid on x_resid (scalar regression)
    xrxr = jnp.dot(x_resid, x_resid)
    xryr = jnp.dot(x_resid, y_resid)
    coef = xryr / jnp.maximum(xrxr, 1e-20)

    # Sum of squares
    ss_null = jnp.dot(y_resid, y_resid)          # RSS of null model
    ss_reduction = coef * xryr                     # SS explained by x
    ss_full = ss_null - ss_reduction               # RSS of full model

    df_full = n - p_null - 1
    ms_full = ss_full / jnp.maximum(df_full, 1.0)

    # F-statistic: (SS_null - SS_full) / 1 / MSE_full
    f_stat = ss_reduction / jnp.maximum(ms_full, 1e-20)

    return coef, f_stat, df_full


# vmap over features (columns of X)
_anova_ftest_batch = jax.vmap(
    _anova_ftest_single, in_axes=(1, None, None, None, None), out_axes=(0, 0, None)
)


@jax.jit
def _residual_ftest_single(x, y, M_null, n, p_null):
    """Closed-form residual F-test for a single feature.

    Compares residual variance of null vs full model.
    """
    x_resid = M_null @ x
    y_resid = M_null @ y

    xrxr = jnp.dot(x_resid, x_resid)
    xryr = jnp.dot(x_resid, y_resid)
    coef = xryr / jnp.maximum(xrxr, 1e-20)

    ss_null = jnp.dot(y_resid, y_resid)
    ss_reduction = coef * xryr
    ss_full = ss_null - ss_reduction

    df_null = n - p_null
    df_full = n - p_null - 1
    ms_null = ss_null / jnp.maximum(df_null, 1.0)
    ms_full = ss_full / jnp.maximum(df_full, 1.0)

    f_stat = ms_null / jnp.maximum(ms_full, 1e-20)
    return coef, f_stat, df_null, df_full


_residual_ftest_batch = jax.vmap(
    _residual_ftest_single, in_axes=(1, None, None, None, None), out_axes=(0, 0, None, None)
)


def _run_anova_test(
    X: jnp.ndarray,
    cond: jnp.ndarray,
    covars: jnp.ndarray,
    method: str = "anova",
    maxiter: int = 100,  # kept for API compat, unused
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run closed-form ANOVA F-tests for a batch of features.

    Parameters
    ----------
    X : (n_samples, n_features) expression data
    cond : (n_samples,) condition/response variable
    covars : (n_samples, n_covariates) covariate matrix including intercept
    method : 'anova' or 'residual'
    """
    n = X.shape[0]
    p_null = covars.shape[1]

    # Precompute null-model projection (shared across all genes)
    M_null = _anova_precompute(covars)

    if method == "anova":
        coefs, f_stat, df_full = _anova_ftest_batch(X, cond, M_null, n, p_null)
        pvals = stats.f.sf(np.asarray(f_stat), 1, int(df_full))
    else:  # residual
        coefs, f_stat, df_null, df_full = _residual_ftest_batch(X, cond, M_null, n, p_null)
        p_resid_cdf = stats.f.cdf(np.asarray(f_stat), int(df_null), int(df_full))
        pvals = 1 - np.abs(0.5 - p_resid_cdf) * 2

    return coefs, np.asarray(f_stat), pvals


def _run_batched_de(
    X: np.ndarray | sparse.spmatrix,
    model_data: pd.DataFrame,
    feature_names: pd.Index,
    method: str,
    condition_key: str,
    dispersions: np.ndarray | None = None,
    size_factors: np.ndarray | None = None,
    covariate_keys: list[str] | None = None,
    design_matrix: np.ndarray | None = None,
    test_idx: int | None = None,
    batch_size: int = 256,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run differential expression analysis in batches.

    This function is the main entry point for performing differential expression
    analysis using JAX-based implementations. It processes large expression matrices
    in batches to optimize memory usage and leverages JAX for acceleration.
    The function supports different statistical methods and handles various
    modeling approaches including offset terms for size factor normalization.

    Parameters
    ----------
    X : np.ndarray | sparse.spmatrix
        Expression data matrix of shape (n_samples, n_features).
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates.
    feature_names : pd.Index
        Names of features/genes corresponding to columns in X.
    method : str
        Statistical method for testing:
        - 'lr': Logistic regression with likelihood ratio test
        - 'anova': Linear model with ANOVA F-test
        - 'anova_residual': Linear model with residual F-test
    condition_key : str
        Name of the column in model_data containing condition labels.
    dispersions : np.ndarray | None, default=None
        Reserved for future use.
    size_factors : np.ndarray | None, default=None
        Reserved for future use.
    covariate_keys : list[str] | None, default=None
        Names of covariate columns in model_data to include in the design matrix.
    design_matrix : np.ndarray | None, default=None
        Pre-built design matrix from formula. When provided, the test column at
        ``test_idx`` is used as the condition and remaining columns as covariates.
        Uses ANOVA (linear model) regardless of ``method``.
    test_idx : int | None, default=None
        Column index in ``design_matrix`` to test. Required when
        ``design_matrix`` is provided.
    batch_size : int, default=32
        Number of features to process in each batch for memory efficiency.
    optimizer : str, default='BFGS'
        Optimization algorithm for fitting models.
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    verbose : bool, default=True
        Whether to display progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with test results for each feature, including:
        - Feature names
        - Coefficients/effect sizes
        - P-values
        - Other test-specific statistics
    """
    # Formula-based path: pre-built design matrix
    if design_matrix is not None:
        if test_idx is None:
            raise ValueError("test_idx is required when design_matrix is provided")
        # Split design into test column and remaining covariates
        test_col = jnp.asarray(design_matrix[:, test_idx], dtype=jnp.float64)
        reduced_cols = [i for i in range(design_matrix.shape[1]) if i != test_idx]
        covars = jnp.asarray(design_matrix[:, reduced_cols], dtype=jnp.float64)

        def test_fn(x):
            return _run_anova_test(x, test_col, covars, "anova", maxiter=maxiter)

    # Prepare data for logistic regression
    elif method == "lr":
        conditions = jnp.asarray(model_data[condition_key].values, dtype=jnp.float64)
        covars = patsy.dmatrix(" + ".join(covariate_keys), model_data) if covariate_keys else np.ones((X.shape[0], 1))
        covars = jnp.asarray(covars, dtype=jnp.float64)

        def test_fn(x):
            return _run_lr_test(x, conditions, covars, optimizer=optimizer, maxiter=maxiter)

    # Prepare data for ANOVA tests
    elif method in ["anova", "anova_residual"]:
        conditions = jnp.asarray(model_data[condition_key].values, dtype=jnp.float64)
        covars = patsy.dmatrix(" + ".join(covariate_keys), model_data) if covariate_keys else np.ones((X.shape[0], 1))
        covars = jnp.asarray(covars, dtype=jnp.float64)
        anova_method = "anova" if method == "anova" else "residual"

        def test_fn(x):
            return _run_anova_test(x, conditions, covars, anova_method, maxiter=maxiter)

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Process run DE tests in batches
    n_features = X.shape[1]
    results = {
        "feature": [],
        "coef": [],
        "stat": [],
        "pval": [],
    }
    for i in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        batch = slice(i, min(i + batch_size, n_features))
        X_batch = jnp.asarray(_to_dense(X[:, batch]), dtype=jnp.float64)
        coefs, test_stats, pvals = test_fn(X_batch)

        results["feature"].extend(feature_names[batch].tolist())
        results["coef"].extend(coefs.tolist())
        results["stat"].extend(np.asarray(test_stats).tolist())
        results["pval"].extend(pvals.tolist())

    return pd.DataFrame(results)
