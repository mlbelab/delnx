"""glmGamPoi-style negative binomial regression models in JAX.

This module implements the core algorithms from the glmGamPoi R package,
adapted for Python with JAX GPU acceleration.

References
----------
Ahlmann-Eltze, C., & Huber, W. (2020). glmGamPoi: fitting Gamma-Poisson
generalized linear models on single cell count data. Bioinformatics.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.special import digamma, gammaln

from ._utils import safe_slogdet

# Enable x64 precision for numerical stability
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Deviance computation
# =============================================================================


@jax.jit
def compute_gp_deviance(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    dispersion: float,
) -> float:
    """Compute Gamma-Poisson (negative binomial) deviance.

    The deviance is defined as 2 * (ll_saturated - ll_model).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape (n_samples,).
    mu : jnp.ndarray
        Fitted mean values, shape (n_samples,).
    dispersion : float
        Overdispersion parameter (theta).

    Returns
    -------
    float
        Total deviance.
    """
    # Clamp values for numerical stability
    counts = jnp.maximum(counts, 0.0)
    mu = jnp.clip(mu, 1e-10, 1e50)
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)

    r = 1.0 / dispersion  # size parameter

    # Log-likelihood for saturated model (mu = counts)
    # For counts = 0, saturated ll is 0
    counts_safe = jnp.maximum(counts, 1e-10)

    ll_saturated = jnp.where(
        counts > 0,
        counts * jnp.log(counts_safe / (counts_safe + r)) + r * jnp.log(r / (r + counts_safe)),
        r * jnp.log(1.0),  # = 0
    )

    # Log-likelihood for fitted model
    ll_model = counts * jnp.log(mu / (mu + r)) + r * jnp.log(r / (r + mu))

    deviance = 2.0 * jnp.sum(ll_saturated - ll_model)
    return deviance


# =============================================================================
# Fisher-scoring beta estimation
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6, 7))
def fit_beta_fisher_scoring(
    counts: jnp.ndarray,
    design: jnp.ndarray,
    offset: jnp.ndarray,
    dispersion: float,
    init_beta: jnp.ndarray,
    maxiter: int = 100,
    tol: float = 1e-8,
    max_line_search: int = 30,
) -> tuple[jnp.ndarray, float, bool]:
    """Fit negative binomial GLM coefficients using Fisher-scoring.

    This implements the Fisher-scoring algorithm with line search,
    following the glmGamPoi approach for robust convergence.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape (n_samples,).
    design : jnp.ndarray
        Design matrix, shape (n_samples, n_coefficients).
    offset : jnp.ndarray
        Offset term (typically log(size_factors)), shape (n_samples,).
    dispersion : float
        Overdispersion parameter.
    init_beta : jnp.ndarray
        Initial coefficient estimates, shape (n_coefficients,).
    maxiter : int, default=100
        Maximum number of Fisher-scoring iterations.
    tol : float, default=1e-8
        Convergence tolerance for relative change in deviance.
    max_line_search : int, default=30
        Maximum number of line search halvings.

    Returns
    -------
    tuple[jnp.ndarray, float, bool]
        - beta: Fitted coefficients, shape (n_coefficients,).
        - deviance: Final deviance.
        - converged: Whether the algorithm converged.
    """
    n_samples, n_coef = design.shape
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)
    r = 1.0 / dispersion  # size parameter

    def compute_mu(beta):
        eta = design @ beta + offset
        eta = jnp.clip(eta, -50, 50)
        return jnp.exp(eta)

    def fisher_step(state):
        """Single Fisher-scoring iteration with line search."""
        i, beta, deviance, converged = state

        mu = compute_mu(beta)
        mu = jnp.clip(mu, 1e-50, 1e50)

        # Working weights: W = mu^2 / V where V = mu + disp * mu^2
        # For NB: W = mu / (1 + disp * mu)
        W = mu / (1.0 + dispersion * mu)
        W = jnp.clip(W, 1e-10, 1e10)

        # Working residuals: z = (y - mu) / mu (for log link)
        residuals = (counts - mu) / mu

        # Score: X^T W (y - mu) / mu * mu = X^T W (y - mu)
        # But we use working response formulation
        # z_working = eta + (y - mu) / mu
        eta = design @ beta + offset
        z_working = eta + residuals

        # Solve weighted least squares: (X^T W X) delta = X^T W z
        W_sqrt = jnp.sqrt(W)
        X_weighted = design * W_sqrt[:, None]
        z_weighted = z_working * W_sqrt

        # QR decomposition for numerical stability
        Q, R = jnp.linalg.qr(X_weighted)
        beta_new = jsp.linalg.solve_triangular(R, Q.T @ z_weighted)

        # Line search to ensure deviance decreases
        def line_search_step(ls_state):
            ls_i, step_size, beta_ls, dev_ls, improved = ls_state
            beta_trial = beta + step_size * (beta_new - beta)
            mu_trial = compute_mu(beta_trial)
            mu_trial = jnp.clip(mu_trial, 1e-50, 1e50)
            dev_trial = compute_gp_deviance(counts, mu_trial, dispersion)

            # Check if deviance decreased
            improved = dev_trial < deviance
            # If not improved, halve step size
            step_size_next = jnp.where(improved, step_size, step_size * 0.5)
            beta_next = jnp.where(improved, beta_trial, beta_ls)
            dev_next = jnp.where(improved, dev_trial, dev_ls)

            return (ls_i + 1, step_size_next, beta_next, dev_next, improved)

        def line_search_cond(ls_state):
            ls_i, _, _, _, improved = ls_state
            return jnp.logical_and(ls_i < max_line_search, ~improved)

        # Initialize line search
        ls_init = (0, 1.0, beta, deviance, False)
        ls_final = jax.lax.while_loop(line_search_cond, line_search_step, ls_init)
        _, _, beta_final, dev_final, _ = ls_final

        # Check convergence
        rel_change = jnp.abs(deviance - dev_final) / (jnp.abs(deviance) + 0.1)
        converged = rel_change < tol

        return (i + 1, beta_final, dev_final, converged)

    def fisher_cond(state):
        i, _, _, converged = state
        return jnp.logical_and(i < maxiter, ~converged)

    # Initialize
    mu_init = compute_mu(init_beta)
    mu_init = jnp.clip(mu_init, 1e-50, 1e50)
    dev_init = compute_gp_deviance(counts, mu_init, dispersion)

    init_state = (0, init_beta, dev_init, False)
    final_state = jax.lax.while_loop(fisher_cond, fisher_step, init_state)
    _, beta_final, deviance_final, converged = final_state

    return beta_final, deviance_final, converged


# Batched version for multiple genes
fit_beta_fisher_scoring_batch = jax.vmap(
    fit_beta_fisher_scoring,
    in_axes=(1, None, None, 0, 1, None, None, None),
    out_axes=(1, 0, 0),
)


# =============================================================================
# One-group fast path (Newton-Raphson for intercept-only models)
# =============================================================================


@jax.jit
def fit_beta_one_group(
    counts: jnp.ndarray,
    size_factors: jnp.ndarray,
    dispersion: float,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float, bool]:
    """Fit intercept-only NB model using Newton-Raphson.

    This is a fast path for single-group designs (no covariates).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape (n_samples,).
    size_factors : jnp.ndarray
        Size factors, shape (n_samples,).
    dispersion : float
        Overdispersion parameter.
    maxiter : int, default=100
        Maximum iterations.
    tol : float, default=1e-8
        Convergence tolerance.

    Returns
    -------
    tuple[float, float, bool]
        - beta0: Fitted intercept (log scale).
        - deviance: Final deviance.
        - converged: Whether converged.
    """
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)
    r = 1.0 / dispersion

    # Initialize with log of normalized mean
    norm_counts = counts / size_factors
    beta0_init = jnp.log(jnp.maximum(jnp.mean(norm_counts), 1e-10))

    def newton_step(state):
        i, beta0, converged = state

        # mu = size_factors * exp(beta0)
        mu = size_factors * jnp.exp(beta0)
        mu = jnp.clip(mu, 1e-50, 1e50)

        # Score (first derivative of log-likelihood w.r.t. beta0)
        # dl/dbeta0 = sum((y - mu) / (1 + disp * mu))
        score = jnp.sum((counts - mu) / (1.0 + dispersion * mu))

        # Fisher information (negative expected second derivative)
        # I = sum(mu / (1 + disp * mu))
        fisher_info = jnp.sum(mu / (1.0 + dispersion * mu))
        fisher_info = jnp.maximum(fisher_info, 1e-10)

        # Newton update
        beta0_new = beta0 + score / fisher_info

        # Check convergence
        converged = jnp.abs(beta0_new - beta0) < tol

        return (i + 1, beta0_new, converged)

    def newton_cond(state):
        i, _, converged = state
        return jnp.logical_and(i < maxiter, ~converged)

    init_state = (0, beta0_init, False)
    final_state = jax.lax.while_loop(newton_cond, newton_step, init_state)
    _, beta0_final, converged = final_state

    # Compute final deviance
    mu_final = size_factors * jnp.exp(beta0_final)
    mu_final = jnp.clip(mu_final, 1e-50, 1e50)
    deviance = compute_gp_deviance(counts, mu_final, dispersion)

    return beta0_final, deviance, converged


# Batched version
fit_beta_one_group_batch = jax.vmap(
    fit_beta_one_group,
    in_axes=(1, None, 0, None, None),
    out_axes=(0, 0, 0),
)


# =============================================================================
# Dispersion estimation with frequency table optimization
# =============================================================================


def _create_frequency_table(counts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create frequency table for count data.

    Parameters
    ----------
    counts : jnp.ndarray
        Count vector, shape (n_samples,).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - unique_counts: Unique count values.
        - frequencies: Frequency of each unique count.
    """
    # This needs to be done in numpy as JAX doesn't support dynamic shapes
    import numpy as np

    counts_np = np.asarray(counts)
    unique_counts, frequencies = np.unique(counts_np, return_counts=True)
    return jnp.array(unique_counts), jnp.array(frequencies)


@jax.jit
def _nb_nll_frequency_table(
    unique_counts: jnp.ndarray,
    frequencies: jnp.ndarray,
    mu: jnp.ndarray,
    dispersion: float,
) -> float:
    """Compute NB negative log-likelihood using frequency table.

    This is the key optimization from glmGamPoi - for sparse data with
    many repeated values (especially zeros), we only compute expensive
    special functions for unique values.

    Parameters
    ----------
    unique_counts : jnp.ndarray
        Unique count values.
    frequencies : jnp.ndarray
        Frequency of each unique count.
    mu : jnp.ndarray
        Mean estimates (full length, n_samples).
    dispersion : float
        Overdispersion parameter.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)
    r = 1.0 / dispersion

    # For frequency table, we approximate mu as the mean
    mu_mean = jnp.mean(mu)
    mu_mean = jnp.clip(mu_mean, 1e-10, 1e50)

    # Compute log-likelihood terms only for unique values
    # ll = gammaln(y + r) - gammaln(y + 1) - gammaln(r) + r*log(r/(r+mu)) + y*log(mu/(r+mu))
    ll_terms = (
        gammaln(unique_counts + r)
        - gammaln(unique_counts + 1)
        - gammaln(r)
        + r * jnp.log(r / (r + mu_mean))
        + unique_counts * jnp.log(mu_mean / (r + mu_mean))
    )

    # Weight by frequencies
    total_ll = jnp.sum(ll_terms * frequencies)

    return -total_ll


@jax.jit
def _nb_nll_full(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    dispersion: float,
) -> float:
    """Compute full NB negative log-likelihood."""
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)
    mu = jnp.clip(mu, 1e-10, 1e50)
    r = 1.0 / dispersion

    ll = (
        gammaln(counts + r)
        - gammaln(counts + 1)
        - gammaln(r)
        + r * jnp.log(r / (r + mu))
        + counts * jnp.log(mu / (r + mu))
    )

    return -jnp.sum(ll)


@jax.jit
def _cox_reid_adjustment(
    design: jnp.ndarray,
    mu: jnp.ndarray,
    dispersion: float,
) -> float:
    """Compute Cox-Reid adjustment term for dispersion estimation.

    Parameters
    ----------
    design : jnp.ndarray
        Design matrix, shape (n_samples, n_coef).
    mu : jnp.ndarray
        Fitted means, shape (n_samples,).
    dispersion : float
        Overdispersion parameter.

    Returns
    -------
    float
        Cox-Reid adjustment term (to be added to negative log-likelihood).
    """
    # W = diag(1 / (1/mu + dispersion))
    W = 1.0 / (1.0 / mu + dispersion)
    W = jnp.clip(W, 1e-10, 1e10)

    # Compute X^T W X
    XtWX = design.T @ (design * W[:, None])

    # Cox-Reid term: 0.5 * log(det(X^T W X))
    _, logdet = safe_slogdet(XtWX)

    return 0.5 * logdet


@partial(jax.jit, static_argnums=(4,))
def estimate_dispersion_mle(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    design: jnp.ndarray,
    init_dispersion: float,
    do_cox_reid: bool = True,
) -> tuple[float, bool]:
    """Estimate dispersion using MLE with optional Cox-Reid adjustment.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape (n_samples,).
    mu : jnp.ndarray
        Fitted means, shape (n_samples,).
    design : jnp.ndarray
        Design matrix, shape (n_samples, n_coef).
    init_dispersion : float
        Initial dispersion estimate.
    do_cox_reid : bool, default=True
        Whether to apply Cox-Reid adjustment.

    Returns
    -------
    tuple[float, bool]
        - dispersion: Estimated dispersion.
        - converged: Whether optimization converged.
    """
    init_dispersion = jnp.clip(init_dispersion, 1e-10, 1e10)
    log_disp_init = jnp.log(init_dispersion)

    def objective(log_disp):
        disp = jnp.exp(log_disp)
        nll = _nb_nll_full(counts, mu, disp)

        if do_cox_reid:
            cr_term = _cox_reid_adjustment(design, mu, disp)
            nll = nll + cr_term

        return nll

    # Use JAX's BFGS optimizer
    result = jsp.optimize.minimize(
        objective,
        jnp.array([log_disp_init]),
        method="BFGS",
        options={"maxiter": 100},
    )

    dispersion = jnp.exp(result.x[0])
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)

    return dispersion, result.success


# =============================================================================
# Moment-based dispersion initialization
# =============================================================================


@jax.jit
def estimate_dispersion_moments(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
) -> float:
    """Estimate dispersion using method of moments.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape (n_samples,).
    mu : jnp.ndarray
        Fitted means, shape (n_samples,).

    Returns
    -------
    float
        Moment-based dispersion estimate.
    """
    # Var = mu + disp * mu^2
    # disp = (Var - mu) / mu^2
    variance = jnp.var(counts, ddof=1)
    mean_mu = jnp.mean(mu)

    dispersion = (variance - mean_mu) / (mean_mu ** 2)
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)

    return dispersion
