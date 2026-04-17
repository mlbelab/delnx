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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median (matching R's matrixStats::weightedMedian)."""
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_w = weights[sorted_idx]
    cumw = np.cumsum(sorted_w)
    half = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, half)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _weighted_median_vectorized(windows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted median for all windows at once.

    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, window_size).
    weights : np.ndarray
        Shape (window_size,).

    Returns
    -------
    np.ndarray
        Weighted medians, shape (n_windows,).
    """
    # Sort each window independently
    sorted_idx = np.argsort(windows, axis=1)
    sorted_vals = np.take_along_axis(windows, sorted_idx, axis=1)
    sorted_w = weights[sorted_idx]  # broadcast weights via sorted indices

    cumw = np.cumsum(sorted_w, axis=1)
    half = cumw[:, -1:] / 2.0

    # Find first index where cumulative weight >= half
    # mask: True where cumw >= half
    mask = cumw >= half
    # argmax on mask gives first True index per row
    med_idx = np.argmax(mask, axis=1)

    return sorted_vals[np.arange(len(windows)), med_idx]


def loc_median_fit(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.1,
    npoints: int | None = None,
    weighted: bool = True,
) -> np.ndarray:
    """Fit a robust local median trend.

    Matches R's glmGamPoi:::loc_median_fit: sliding window of ``npoints``
    elements with Gaussian-weighted medians, evaluated at every interior
    point with edge-clamping.

    Uses vectorized sliding windows via np.lib.stride_tricks for performance.

    Parameters
    ----------
    x : np.ndarray
        Predictor values (e.g., mean expression), shape (n_genes,).
    y : np.ndarray
        Response values (e.g., dispersion), shape (n_genes,).
    fraction : float, default=0.1
        Fraction of data for the window size (npoints = round(n * fraction)).
    npoints : int | None, default=None
        Window size. If None, computed from fraction.
    weighted : bool, default=True
        Use Gaussian-weighted median (matching R default).

    Returns
    -------
    np.ndarray
        Fitted trend values at each x, shape (n_genes,).
    """
    n = len(x)
    if n == 0:
        return np.array([])

    if npoints is None:
        npoints = max(1, round(n * fraction))
    npoints = max(1, min(npoints, n))

    ordered_indices = np.argsort(x)
    ordered_y = y[ordered_indices]

    half_points = npoints // 2
    window_size = half_points * 2 + 1

    # Gaussian weights for the window
    if weighted:
        weights = stats.norm.pdf(np.linspace(-3, 3, window_size))
    else:
        weights = np.ones(window_size)

    start = half_points
    end = n - 1 - half_points

    if end < start:
        # Window larger than data
        if weighted:
            w = stats.norm.pdf(np.linspace(-3, 3, n))
            wm = _weighted_median(ordered_y, w)
        else:
            wm = np.median(ordered_y)
        return np.full(n, wm)

    # Build all windows at once using stride_tricks
    # This creates a view (no copy) of shape (n_windows, window_size)
    n_windows = end - start + 1
    windows = np.lib.stride_tricks.as_strided(
        ordered_y[start - half_points:],
        shape=(n_windows, window_size),
        strides=(ordered_y.strides[0], ordered_y.strides[0]),
    ).copy()  # copy to avoid issues with non-contiguous memory in argsort

    # Compute all weighted medians vectorized
    if weighted:
        medians = _weighted_median_vectorized(windows, weights)
    else:
        medians = np.median(windows, axis=1)

    res = np.empty(n)
    res[start:end + 1] = medians

    # Edge clamping
    res[:start] = res[start]
    res[end + 1:] = res[end]

    # Unsort: map back to original order
    result = np.empty(n)
    result[ordered_indices] = res
    return result


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
        return np.full_like(dispersions, np.mean(dispersions[valid_mask]))

    if method == "mean":
        return np.full_like(dispersions, np.mean(dispersions[valid_mask]))

    elif method == "local_median":
        # R calls: loc_median_fit(gene_means[valid], y=disp_est[valid])
        # Directly on the raw scale, not log-log
        trend = np.full_like(dispersions, np.nan)
        trend[valid_mask] = loc_median_fit(
            mean_expression[valid_mask], dispersions[valid_mask]
        )

        # Fill invalid genes with nearest valid value
        if (~valid_mask).any():
            from scipy.interpolate import interp1d

            valid_idx = np.where(valid_mask)[0]
            invalid_idx = np.where(~valid_mask)[0]
            fill_fn = interp1d(
                mean_expression[valid_mask],
                trend[valid_mask],
                kind="nearest",
                bounds_error=False,
                fill_value=(trend[valid_mask][0], trend[valid_mask][-1]),
            )
            trend[invalid_idx] = fill_fn(mean_expression[invalid_idx])

        return trend

    else:
        raise ValueError(f"Unknown trend method: {method}")


# =============================================================================
# Empirical Bayes shrinkage
# =============================================================================


def _estimate_prior_df(
    ql_dispersions: np.ndarray,
    df_residual: int,
) -> tuple[float, float]:
    """Estimate prior degrees of freedom via F-distribution MLE.

    Matches R's glmGamPoi:::variance_prior: jointly estimates variance0
    and df0 by maximizing the F-distribution log-likelihood of
    ``s2 / variance0 ~ F(df_residual, df0)``.

    Parameters
    ----------
    ql_dispersions : np.ndarray
        Quasi-likelihood dispersion estimates, shape (n_genes,).
    df_residual : int
        Residual degrees of freedom from the GLM fit.

    Returns
    -------
    tuple[float, float]
        - df0: Prior degrees of freedom.
        - s0_sq: Prior variance estimate (variance0).
    """
    valid = np.isfinite(ql_dispersions) & (ql_dispersions > 0)
    s2 = ql_dispersions[valid]

    if len(s2) < 10:
        return 0.0, 1.0

    if np.all(s2 == 1.0):
        return np.inf, 1.0

    # Pre-compute log(s2) once — used in every evaluation
    log_s2 = np.log(s2)
    n = len(s2)
    d1 = float(df_residual)

    # Analytical F-distribution log-PDF (avoids scipy.stats.f overhead):
    # log f(x; d1, d2) = 0.5*[d1*log(d1) + d2*log(d2)] + 0.5*(d1-2)*log(x)
    #                   - 0.5*(d1+d2)*log(d1*x + d2) - log(B(d1/2, d2/2))
    # where B is the beta function.
    from scipy.special import betaln

    def neg_loglik(par):
        log_var0 = par[0]
        d2 = np.exp(np.clip(par[1], -50, 50))

        log_x = log_s2 - log_var0

        half_d1 = 0.5 * d1
        half_d2 = 0.5 * d2
        # F log-PDF (vectorized, no scipy.stats overhead)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_pdf = (
                half_d1 * np.log(d1) + half_d2 * np.log(d2)
                - betaln(half_d1, half_d2)
                + (half_d1 - 1.0) * log_x
                - (half_d1 + half_d2) * np.log(d1 * np.exp(np.clip(log_x, -500, 500)) + d2)
            )
        log_pdf = np.where(np.isfinite(log_pdf), log_pdf, -1e10)
        return -np.sum(log_pdf - log_var0)

    try:
        result = optimize.minimize(
            neg_loglik,
            x0=[0.0, 0.0],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        s0_sq = np.exp(result.x[0])
        df0 = np.exp(result.x[1])
    except Exception:
        # Fallback
        s0_sq = np.median(s2)
        df0 = 10.0

    s0_sq = max(s0_sq, 1e-10)
    df0 = max(df0, 0.1)

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
