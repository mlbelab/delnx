"""glmGamPoi-style negative binomial regression models in JAX.

This module implements the core algorithms from the glmGamPoi R package,
adapted for Python with JAX GPU acceleration. All core solvers use
jax.lax.while_loop for full JIT-ability and are vmapped for batch
processing across genes.

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


# Batched deviance: genes along axis 1 of counts/mu, dispersions is (n_genes,)
compute_gp_deviance_batch = jax.jit(jax.vmap(
    compute_gp_deviance,
    in_axes=(1, 1, 0),
    out_axes=0,
))


# =============================================================================
# Newton-Raphson beta estimation (fully JIT-able via lax.while_loop)
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def fit_beta_newton(
    counts: jnp.ndarray,
    design: jnp.ndarray,
    offset: jnp.ndarray,
    dispersion: float,
    init_beta: jnp.ndarray,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> tuple[jnp.ndarray, float, bool]:
    """Fit NB GLM coefficients using Newton-Raphson with observed Hessian.

    Fully JIT-able implementation using jax.lax.while_loop, enabling
    vmap across genes for batch processing.

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
        Maximum number of Newton-Raphson iterations.
    tol : float, default=1e-8
        Convergence tolerance for maximum absolute change in beta.

    Returns
    -------
    tuple[jnp.ndarray, float, bool]
        - beta: Fitted coefficients, shape (n_coefficients,).
        - deviance: Final deviance.
        - converged: Whether the algorithm converged.
    """
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)

    def newton_body(state):
        i, beta, converged = state

        eta = design @ beta + offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)
        mu = jnp.clip(mu, 1e-10, 1e50)

        # Score (gradient of log-likelihood)
        score = design.T @ ((counts - mu) / (1.0 + dispersion * mu))

        # Observed Hessian: W_obs_i = mu_i * (1 + disp * y_i) / (1 + disp * mu_i)^2
        W_obs = mu * (1.0 + dispersion * counts) / (1.0 + dispersion * mu) ** 2
        W_obs = jnp.clip(W_obs, 1e-10, 1e10)

        H = design.T @ (design * W_obs[:, None])
        H = H + jnp.eye(H.shape[0]) * 1e-6

        delta = jnp.linalg.solve(H, score)

        # Clamp step size
        max_step = jnp.max(jnp.abs(delta))
        scale = jnp.where(max_step > 5.0, 5.0 / max_step, 1.0)
        delta = delta * scale

        beta_new = beta + delta
        converged = jnp.max(jnp.abs(delta)) < tol
        return (i + 1, beta_new, converged)

    def newton_cond(state):
        i, _, converged = state
        return jnp.logical_and(i < maxiter, ~converged)

    init_state = (0, init_beta, False)
    _, beta_final, converged = jax.lax.while_loop(newton_cond, newton_body, init_state)

    # Compute final deviance
    eta = design @ beta_final + offset
    eta = jnp.clip(eta, -50, 50)
    mu_final = jnp.exp(eta)
    mu_final = jnp.clip(mu_final, 1e-50, 1e50)
    deviance_final = compute_gp_deviance(counts, mu_final, dispersion)

    return beta_final, deviance_final, converged


# Legacy alias for backward compatibility
def fit_beta_fisher_scoring(
    counts: jnp.ndarray,
    design: jnp.ndarray,
    offset: jnp.ndarray,
    dispersion: float,
    init_beta: jnp.ndarray,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> tuple[jnp.ndarray, float, bool]:
    """Fit NB GLM coefficients (legacy wrapper for fit_beta_newton)."""
    return fit_beta_newton(counts, design, offset, dispersion, init_beta, maxiter, tol)


# Batched version: vmap across genes
# counts: (n_samples, n_genes) -> vmap over axis 1
# design: shared (n_samples, n_coef) -> not vmapped
# offset: shared (n_samples,) -> not vmapped
# dispersion: (n_genes,) -> vmap over axis 0
# init_beta: (n_genes, n_coef) -> vmap over axis 0
fit_beta_newton_batch = jax.jit(jax.vmap(
    fit_beta_newton,
    in_axes=(1, None, None, 0, 0, None, None),
    out_axes=(0, 0, 0),
), static_argnums=(5, 6))


# =============================================================================
# One-group fast path (Newton-Raphson for intercept-only models)
# =============================================================================


@partial(jax.jit, static_argnums=(3, 4))
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
fit_beta_one_group_batch = jax.jit(jax.vmap(
    fit_beta_one_group,
    in_axes=(1, None, 0, None, None),
    out_axes=(0, 0, 0),
), static_argnums=(3, 4))


# =============================================================================
# Dispersion estimation
# =============================================================================


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


@partial(jax.jit, static_argnums=(4, 5, 6))
def estimate_dispersion_mle_newton(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    design: jnp.ndarray,
    init_dispersion: float,
    do_cox_reid: bool = True,
    maxiter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, bool]:
    """Estimate dispersion using Newton's method on log-dispersion.

    Fully JIT-able 1D Newton optimizer on the NLL w.r.t. log(dispersion),
    using JAX autodiff for gradient and Hessian. Replaces the BFGS optimizer
    for vmap compatibility.

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
    maxiter : int, default=50
        Maximum iterations.
    tol : float, default=1e-6
        Convergence tolerance on log-dispersion change.

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
        cr = jnp.where(
            do_cox_reid,
            _cox_reid_adjustment(design, mu, disp),
            0.0,
        )
        return nll + cr

    grad_fn = jax.grad(objective)
    hess_fn = jax.grad(grad_fn)

    def newton_body(state):
        i, log_disp, converged = state
        g = grad_fn(log_disp)
        h = hess_fn(log_disp)
        # Ensure positive curvature
        h = jnp.maximum(h, 1e-6)
        step = -g / h
        # Clamp step size
        step = jnp.clip(step, -2.0, 2.0)

        # Simple backtracking: halve step if it increases the objective
        log_disp_new = jnp.clip(log_disp + step, -23.0, 23.0)
        obj_old = objective(log_disp)
        obj_new = objective(log_disp_new)

        # If no improvement, try half step, then quarter step
        log_disp_half = jnp.clip(log_disp + step * 0.5, -23.0, 23.0)
        obj_half = objective(log_disp_half)
        log_disp_new = jnp.where(obj_half < obj_new, log_disp_half, log_disp_new)
        obj_new = jnp.minimum(obj_half, obj_new)

        log_disp_quarter = jnp.clip(log_disp + step * 0.25, -23.0, 23.0)
        obj_quarter = objective(log_disp_quarter)
        log_disp_new = jnp.where(obj_quarter < obj_new, log_disp_quarter, log_disp_new)

        converged = jnp.abs(log_disp_new - log_disp) < tol
        return (i + 1, log_disp_new, converged)

    def newton_cond(state):
        i, _, converged = state
        return jnp.logical_and(i < maxiter, ~converged)

    _, log_disp_final, converged = jax.lax.while_loop(
        newton_cond, newton_body, (0, log_disp_init, False)
    )

    dispersion = jnp.exp(log_disp_final)
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)

    return dispersion, converged


# Batched version: vmap across genes
# counts: (n_samples, n_genes) -> axis 1
# mu: (n_samples, n_genes) -> axis 1
# design: shared -> None
# init_dispersion: (n_genes,) -> axis 0
estimate_dispersion_mle_batch = jax.jit(jax.vmap(
    estimate_dispersion_mle_newton,
    in_axes=(1, 1, None, 0, None, None, None),
    out_axes=(0, 0),
), static_argnums=(4, 5, 6))


# Legacy BFGS wrapper (kept for single-gene use and backward compat)
@partial(jax.jit, static_argnums=(4,))
def estimate_dispersion_mle(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    design: jnp.ndarray,
    init_dispersion: float,
    do_cox_reid: bool = True,
) -> tuple[float, bool]:
    """Estimate dispersion using MLE (BFGS, single-gene legacy interface)."""
    return estimate_dispersion_mle_newton(
        counts, mu, design, init_dispersion, do_cox_reid
    )


# =============================================================================
# Moment-based dispersion initialization
# =============================================================================


@jax.jit
def _estimate_dispersion_moments_single(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
) -> float:
    """Estimate dispersion using method of moments (single gene)."""
    variance = jnp.var(counts, ddof=1)
    mean_mu = jnp.mean(mu)
    dispersion = (variance - mean_mu) / (mean_mu ** 2)
    dispersion = jnp.clip(dispersion, 1e-10, 1e10)
    return dispersion


# Legacy alias
estimate_dispersion_moments = _estimate_dispersion_moments_single

# Batched: counts (n_samples, n_genes), mu (n_samples, n_genes)
estimate_dispersion_moments_batch = jax.jit(jax.vmap(
    _estimate_dispersion_moments_single,
    in_axes=(1, 1),
    out_axes=0,
))
