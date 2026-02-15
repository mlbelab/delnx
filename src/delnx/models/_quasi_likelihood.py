"""Quasi-likelihood framework for glmGamPoi-style differential expression.

This module implements quasi-likelihood dispersion estimation and empirical
Bayes shrinkage following the approach in glmGamPoi and limma/edgeR.

References
----------
- Lund et al. (2012). Detecting Differential Expression in RNA-sequence Data
  Using Quasi-likelihood with Shrunken Dispersion Estimates. SAGMB.
- Smyth (2004). Linear Models and Empirical Bayes Methods for Assessing
  Differential Expression in Microarray Experiments. SAGMB.
"""

import numpy as np
from scipy import optimize, stats
from scipy.special import polygamma


# =============================================================================
# Quasi-likelihood dispersion transformation
# =============================================================================


def compute_ql_dispersions(
    dispersions_mle: np.ndarray,
    mu_means: np.ndarray,
    dispersion_trend: np.ndarray,
) -> np.ndarray:
    """Transform MLE dispersions to quasi-likelihood scale.

    The QL dispersion captures the ratio of observed to expected variance
    under the fitted dispersion trend.

    Parameters
    ----------
    dispersions_mle : np.ndarray
        MLE dispersion estimates per gene, shape (n_genes,).
    mu_means : np.ndarray
        Mean expression per gene, shape (n_genes,).
    dispersion_trend : np.ndarray
        Fitted dispersion trend values, shape (n_genes,).

    Returns
    -------
    np.ndarray
        Quasi-likelihood dispersion estimates, shape (n_genes,).
    """
    # QL dispersion: ratio of observed to expected variance
    # ql_disp = (1 + mu * disp_mle) / (1 + mu * disp_trend)
    mu_means = np.maximum(mu_means, 1e-10)
    dispersion_trend = np.maximum(dispersion_trend, 1e-10)
    dispersions_mle = np.maximum(dispersions_mle, 1e-10)

    observed_var_factor = 1.0 + mu_means * dispersions_mle
    expected_var_factor = 1.0 + mu_means * dispersion_trend

    ql_dispersions = observed_var_factor / expected_var_factor

    # Ensure positive
    ql_dispersions = np.maximum(ql_dispersions, 1e-10)

    return ql_dispersions


# =============================================================================
# Local median fit for dispersion trend
# =============================================================================


def loc_median_fit(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.3,
    n_bins: int = 50,
) -> np.ndarray:
    """Fit a robust local median trend.

    This is a simplified version of the local median fit used in glmGamPoi
    for estimating dispersion trends. It's more robust to outliers than loess.

    Parameters
    ----------
    x : np.ndarray
        Predictor values (e.g., log mean expression), shape (n_genes,).
    y : np.ndarray
        Response values (e.g., log dispersion), shape (n_genes,).
    fraction : float, default=0.3
        Fraction of data to use in each local window.
    n_bins : int, default=50
        Number of bins for computing local medians.

    Returns
    -------
    np.ndarray
        Fitted trend values at each x, shape (n_genes,).
    """
    n = len(x)

    # Handle edge cases
    if n < 10:
        return np.full(n, np.median(y))

    # Create bins based on x values
    x_sorted_idx = np.argsort(x)
    x_sorted = x[x_sorted_idx]
    y_sorted = y[x_sorted_idx]

    # Compute bin edges
    bin_edges = np.linspace(0, n - 1, n_bins + 1).astype(int)

    # Compute median for each bin
    bin_centers = np.zeros(n_bins)
    bin_medians = np.zeros(n_bins)

    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1] if i < n_bins - 1 else n
        if end > start:
            bin_centers[i] = np.median(x_sorted[start:end])
            bin_medians[i] = np.median(y_sorted[start:end])
        else:
            bin_centers[i] = x_sorted[start] if start < n else x_sorted[-1]
            bin_medians[i] = y_sorted[start] if start < n else y_sorted[-1]

    # Interpolate to get trend at each original x
    trend = np.interp(x, bin_centers, bin_medians)

    return trend


def fit_dispersion_trend(
    dispersions: np.ndarray,
    mean_expression: np.ndarray,
    method: str = "local_median",
) -> np.ndarray:
    """Fit a trend to dispersion estimates.

    Parameters
    ----------
    dispersions : np.ndarray
        MLE dispersion estimates, shape (n_genes,).
    mean_expression : np.ndarray
        Mean expression per gene, shape (n_genes,).
    method : str, default="local_median"
        Method for fitting trend. Options: "local_median", "mean".

    Returns
    -------
    np.ndarray
        Fitted trend values, shape (n_genes,).
    """
    # Filter out extreme values
    valid_mask = (dispersions > 1e-10) & (mean_expression > 1e-10)
    valid_mask &= np.isfinite(dispersions) & np.isfinite(mean_expression)

    if valid_mask.sum() < 10:
        # Not enough valid genes, return mean
        return np.full_like(dispersions, np.median(dispersions[valid_mask]))

    if method == "mean":
        return np.full_like(dispersions, np.median(dispersions[valid_mask]))

    elif method == "local_median":
        # Fit in log-log space
        log_mean = np.log10(mean_expression[valid_mask])
        log_disp = np.log10(dispersions[valid_mask])

        # Fit local median
        log_trend_valid = loc_median_fit(log_mean, log_disp)

        # Interpolate back to all genes
        log_mean_all = np.log10(np.maximum(mean_expression, 1e-10))
        log_trend_all = np.interp(
            log_mean_all,
            log_mean[np.argsort(log_mean)],
            log_trend_valid[np.argsort(log_mean)],
        )

        return 10 ** log_trend_all

    else:
        raise ValueError(f"Unknown trend method: {method}")


# =============================================================================
# Empirical Bayes shrinkage
# =============================================================================


def _estimate_prior_df(
    ql_dispersions: np.ndarray,
    df_residual: int,
    robust: bool = True,
) -> tuple[float, float]:
    """Estimate prior degrees of freedom for inverse chi-square distribution.

    This implements the empirical Bayes estimation of the prior distribution
    parameters for quasi-likelihood dispersions.

    Parameters
    ----------
    ql_dispersions : np.ndarray
        Quasi-likelihood dispersion estimates, shape (n_genes,).
    df_residual : int
        Residual degrees of freedom from the GLM fit.
    robust : bool, default=True
        Whether to use robust estimation (winsorization).

    Returns
    -------
    tuple[float, float]
        - df0: Prior degrees of freedom.
        - s0_sq: Prior variance estimate.
    """
    # Filter valid dispersions
    valid = np.isfinite(ql_dispersions) & (ql_dispersions > 0)
    ql_valid = ql_dispersions[valid]

    if len(ql_valid) < 10:
        # Not enough genes, return uninformative prior
        return 0.0, 1.0

    # Robust estimation: winsorize extreme values
    if robust:
        lower = np.percentile(ql_valid, 1)
        upper = np.percentile(ql_valid, 99)
        ql_valid = np.clip(ql_valid, lower, upper)

    # Method of moments for inverse chi-square
    # E[s^2] = s0^2 * df0 / (df0 - 2) for df0 > 2
    # Var[s^2] = 2 * s0^4 * df0^2 / ((df0-2)^2 * (df0-4)) for df0 > 4

    mean_s2 = np.mean(ql_valid)
    var_s2 = np.var(ql_valid, ddof=1)

    # Estimate df0 using trigamma function
    # Based on limma::fitFDist approach
    log_ql = np.log(ql_valid)
    mean_log = np.mean(log_ql)

    # E[log(s^2)] = log(s0^2) + digamma(df0/2) - log(df0/2)
    # Var[log(s^2)] = trigamma(df0/2)

    var_log = np.var(log_ql, ddof=1)

    # Solve for df0 using trigamma inverse
    # trigamma(df0/2) = var_log
    # Use numerical optimization
    def objective(df0):
        if df0 <= 0:
            return 1e10
        return (polygamma(1, df0 / 2) - var_log) ** 2

    try:
        result = optimize.minimize_scalar(
            objective,
            bounds=(0.1, 1000),
            method="bounded",
        )
        df0 = result.x
    except Exception:
        df0 = 10.0  # Default fallback

    # Ensure df0 is reasonable
    df0 = np.clip(df0, 0.1, 1000)

    # Estimate s0^2
    # s0^2 = mean(s^2) * (df0 - 2) / df0 for df0 > 2
    if df0 > 2:
        s0_sq = mean_s2 * (df0 - 2) / df0
    else:
        s0_sq = np.median(ql_valid)

    s0_sq = max(s0_sq, 1e-10)

    return df0, s0_sq


def shrink_ql_dispersions(
    ql_dispersions: np.ndarray,
    df_residual: int,
    df0: float | None = None,
    s0_sq: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply empirical Bayes shrinkage to quasi-likelihood dispersions.

    This implements the posterior estimation under an inverse chi-square prior,
    following the limma/edgeR approach.

    Parameters
    ----------
    ql_dispersions : np.ndarray
        Quasi-likelihood dispersion estimates, shape (n_genes,).
    df_residual : int
        Residual degrees of freedom from GLM fit.
    df0 : float | None, default=None
        Prior degrees of freedom. If None, estimated from data.
    s0_sq : float | None, default=None
        Prior variance. If None, estimated from data.

    Returns
    -------
    tuple[np.ndarray, float, float]
        - shrunken_dispersions: Posterior dispersion estimates, shape (n_genes,).
        - df0: Prior degrees of freedom used.
        - s0_sq: Prior variance used.
    """
    # Estimate prior if not provided
    if df0 is None or s0_sq is None:
        df0_est, s0_sq_est = _estimate_prior_df(ql_dispersions, df_residual)
        df0 = df0_est if df0 is None else df0
        s0_sq = s0_sq_est if s0_sq is None else s0_sq

    # Posterior mean under inverse chi-square prior
    # posterior_var = (df0 * s0^2 + df * s^2) / (df0 + df)
    df_total = df0 + df_residual

    shrunken = (df0 * s0_sq + df_residual * ql_dispersions) / df_total

    # Ensure positive
    shrunken = np.maximum(shrunken, 1e-10)

    return shrunken, df0, s0_sq


# =============================================================================
# Quasi-likelihood F-test
# =============================================================================


def ql_f_test(
    deviance_full: np.ndarray,
    deviance_reduced: np.ndarray,
    df_full: int,
    df_reduced: int,
    ql_dispersions: np.ndarray,
    df0_prior: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform quasi-likelihood F-test for differential expression.

    This tests the null hypothesis that the reduced model is adequate
    compared to the full model.

    Parameters
    ----------
    deviance_full : np.ndarray
        Deviance from full model, shape (n_genes,).
    deviance_reduced : np.ndarray
        Deviance from reduced model, shape (n_genes,).
    df_full : int
        Degrees of freedom for full model (n_samples - n_coef_full).
    df_reduced : int
        Degrees of freedom for reduced model (n_samples - n_coef_reduced).
    ql_dispersions : np.ndarray
        Quasi-likelihood dispersion estimates, shape (n_genes,).
    df0_prior : float, default=0.0
        Prior degrees of freedom from empirical Bayes shrinkage.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - f_statistics: F-test statistics, shape (n_genes,).
        - p_values: P-values from F-distribution, shape (n_genes,).
    """
    # Degrees of freedom for the test
    df_diff = df_reduced - df_full  # Number of parameters being tested
    df_denom = df_full + df0_prior  # Denominator df includes prior

    # Ensure positive deviance differences
    dev_diff = np.maximum(deviance_reduced - deviance_full, 0.0)

    # F statistic
    # F = (Dev_reduced - Dev_full) / (df_diff * ql_disp)
    ql_dispersions = np.maximum(ql_dispersions, 1e-10)
    f_statistics = dev_diff / (df_diff * ql_dispersions)

    # P-values from F-distribution
    p_values = stats.f.sf(f_statistics, df_diff, df_denom)

    # Handle edge cases
    p_values = np.clip(p_values, 0.0, 1.0)
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)

    return f_statistics, p_values


def ql_test_contrast(
    deviance_full: np.ndarray,
    deviance_reduced: np.ndarray,
    n_samples: int,
    n_coef_full: int,
    n_coef_reduced: int,
    ql_dispersions: np.ndarray,
    df0_prior: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for QL F-test with coefficient counts.

    Parameters
    ----------
    deviance_full : np.ndarray
        Deviance from full model, shape (n_genes,).
    deviance_reduced : np.ndarray
        Deviance from reduced model, shape (n_genes,).
    n_samples : int
        Number of samples.
    n_coef_full : int
        Number of coefficients in full model.
    n_coef_reduced : int
        Number of coefficients in reduced model.
    ql_dispersions : np.ndarray
        QL dispersion estimates, shape (n_genes,).
    df0_prior : float, default=0.0
        Prior degrees of freedom.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - f_statistics: F-test statistics.
        - p_values: P-values.
    """
    df_full = n_samples - n_coef_full
    df_reduced = n_samples - n_coef_reduced

    return ql_f_test(
        deviance_full,
        deviance_reduced,
        df_full,
        df_reduced,
        ql_dispersions,
        df0_prior,
    )
