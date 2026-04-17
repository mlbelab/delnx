"""Regression models in JAX."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.optimize as scipy_optimize
from jax.scipy import optimize
from scipy.special import polygamma
from scipy.stats import trim_mean

from ._utils import grid_fit_alpha, mean_absolute_deviation, nb_nll, safe_slogdet

# Enable x64 precision globally
try:
    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "JAX x64 precision could not be enabled. This might lead to numerical instabilities.", stacklevel=2
        )
except Exception as e:  # noqa: BLE001
    warnings.warn(f"JAX configuration failed: {e}", stacklevel=2)


@dataclass(frozen=True)
class Regression:
    """Base class for regression models.

    This is the abstract base class for all regression models in the package.
    It provides common functionality for fitting models, computing statistics,
    and handling offsets for normalization.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options include "BFGS" and "IRLS"
        (Iteratively Reweighted Least Squares) for GLM-type models.
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics (for faster computation).
    """

    maxiter: int = 100
    tol: float = 1e-6
    optimizer: str = "BFGS"
    skip_stats: bool = False

    def _fit_bfgs(self, neg_ll_fn: Callable, init_params: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Fit model using the BFGS optimizer.

        Parameters
        ----------
        neg_ll_fn : Callable
            Function that computes the negative log-likelihood.
        init_params : jnp.ndarray
            Initial parameter values.
        **kwargs
            Additional arguments passed to the optimizer.

        Returns
        -------
        jnp.ndarray
            Optimized parameters.
        """
        result = optimize.minimize(neg_ll_fn, init_params, method="BFGS", options={"maxiter": self.maxiter})
        return result.x

    def _fit_irls(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        weight_fn: Callable,
        working_resid_fn: Callable,
        init_params: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Fit model using Iteratively Reweighted Least Squares algorithm.

        This implements the IRLS algorithm for generalized linear models
        with support for offset terms. For count models (e.g., Negative
        Binomial), the offset is used to incorporate size factors.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        weight_fn : Callable
            Function to compute weights at each iteration.
        working_resid_fn : Callable
            Function to compute working residuals at each iteration.
        init_params : jnp.ndarray
            Initial parameter values.
        offset : jnp.ndarray | None, default=None
            Offset term (log scale for GLMs) to include in the model.
        **kwargs
            Additional arguments passed to weight_fn and working_resid_fn.

        Returns
        -------
        jnp.ndarray
            Optimized parameters.
        """
        n, p = X.shape
        eps = 1e-6

        # Handle offset
        if offset is None:
            offset = jnp.zeros(n)

        def irls_step(state):
            i, converged, beta = state

            # Compute weights and working residuals
            W = weight_fn(X, beta, offset=offset, **kwargs)
            z = working_resid_fn(X, y, beta, offset=offset, **kwargs)

            # Weighted design matrix
            W_sqrt = jnp.sqrt(W)
            X_weighted = X * W_sqrt[:, None]
            z_weighted = z * W_sqrt

            # Solve weighted least squares: (X^T W X) β = X^T W z
            XtWX = X_weighted.T @ X_weighted
            XtWz = X_weighted.T @ z_weighted
            beta_new = jax.scipy.linalg.solve(XtWX + eps * jnp.eye(p), XtWz, assume_a="pos")

            # Check convergence
            delta = jnp.max(jnp.abs(beta_new - beta))
            converged = delta < self.tol

            return i + 1, converged, beta_new

        def irls_cond(state):
            i, converged, _ = state
            return jnp.logical_and(i < self.maxiter, ~converged)

        # Initialize state
        state = (0, False, init_params)
        final_state = jax.lax.while_loop(irls_cond, irls_step, state)
        _, _, beta_final = final_state
        return beta_final

    def _compute_stats(
        self,
        X: jnp.ndarray,
        neg_ll_fn: Callable,
        params: jnp.ndarray,
        test_idx: int = -1,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute test statistics for fitted parameters.
        This method computes the Wald test statistics and p-values for the
        fitted parameters using the Hessian of the negative log-likelihood function.
        If the Hessian is ill-conditioned, it falls back to a likelihood ratio test.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        neg_ll_fn : Callable
            Function that computes the negative log-likelihood.
        params : jnp.ndarray
            Fitted parameter estimates.
        test_idx : int, default=-1
            Index of the parameter to test. If -1, tests the last parameter.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Standard errors, test statistics, and p-values.
        """  # noqa: D205
        hess_fn = jax.hessian(neg_ll_fn)
        hessian = hess_fn(params)
        hessian = 0.5 * (hessian + hessian.T)

        # Check condition number
        condition_number = jnp.linalg.cond(hessian)

        def wald_test():
            """Perform Wald test."""
            se = jnp.sqrt(jnp.clip(jnp.diag(jnp.linalg.inv(hessian)), 1e-8))
            stat = (params / se) ** 2
            pval = jsp.stats.chi2.sf(stat, df=1)
            return se, stat, pval

        def likelihood_ratio_test():
            """Perform likelihood ratio test as a fallback for ill-conditioned cases."""
            ll_full = -neg_ll_fn(params)
            params_reduced = params.at[test_idx].set(0.0)
            ll_reduced = -neg_ll_fn(params_reduced)
            # Compute likelihood ratio statistic
            lr_stat = 2 * (ll_full - ll_reduced)
            lr_stat = jnp.maximum(lr_stat, 0.0)
            # Compute correction for small sample sizes (where appropriate)
            n_samples = X.shape[0]
            n_params = X.shape[1]
            correction = 1 + n_params / jnp.maximum(1.0, n_samples - n_params)
            corrected_lr_stat = lr_stat / correction
            # Compute p-value for the likelihood ratio statistic
            lr_pval = jsp.stats.chi2.sf(corrected_lr_stat, df=1)
            # Return dummy values for SE and stat
            se = jnp.full_like(params, jnp.nan)
            stat = jnp.zeros_like(params)
            stat = stat.at[test_idx].set(lr_stat)
            pval = jnp.ones_like(params)
            pval = pval.at[test_idx].set(lr_pval)
            return se, stat, pval

        stats = jax.lax.cond(
            condition_number < 1e5,  # Relatively conservative threshold
            lambda _: wald_test(),
            lambda _: likelihood_ratio_test(),
            operand=None,
        )

        return stats

    def _exact_solution(self, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute exact Ordinary Least Squares solution.

        For linear regression, the offset is incorporated by adjusting the
        response variable (y - offset) rather than the linear predictor.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        offset : jnp.ndarray | None, default=None
            Offset term to include in the model.

        Returns
        -------
        jnp.ndarray
            Coefficient estimates.
        """
        if offset is not None:
            # Adjust y by subtracting offset for linear regression
            y_adj = y - offset
        else:
            y_adj = y

        XtX = X.T @ X
        Xty = X.T @ y_adj
        params = jax.scipy.linalg.solve(XtX, Xty, assume_a="pos")
        return params

    def get_llf(self, X: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, offset: jnp.ndarray | None = None) -> float:
        """Get log-likelihood at fitted parameters.

        This method converts the negative log-likelihood to a log-likelihood
        value, which is useful for model comparison and likelihood ratio tests.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        params : jnp.ndarray
            Parameter estimates.
        offset : jnp.ndarray | None, default=None
            Offset term to include in the model.

        Returns
        -------
        float
            Log-likelihood value.
        """
        nll = self._negative_log_likelihood(params, X, y, offset)
        return -nll  # Convert negative log-likelihood to log-likelihood


@dataclass(frozen=True)
class LinearRegression(Regression):
    """Linear regression with Ordinary Least Squares estimation.

    This class implements a basic linear regression model using OLS, with support for
    including offset terms. For linear models, offsets are applied by subtracting
    from the response variable rather than adding to the linear predictor.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization (inherited from Regression).
    tol : float, default=1e-6
        Convergence tolerance (inherited from Regression).
    optimizer : str, default="BFGS"
        Optimization method (inherited from Regression).
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics (inherited from Regression).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import LinearRegression
    >>> X = jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])  # Design matrix with intercept
    >>> y = jnp.array([1.0, 2.0, 3.0])  # Response variable
    >>> model = LinearRegression()
    >>> result = model.fit(X, y)
    >>> print(f"Coefficients: {result['coef']}")
    """

    def _negative_log_likelihood(
        self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> float:
        """Compute negative log likelihood (assuming Gaussian noise) with offset."""
        pred = jnp.dot(X, params)
        if offset is not None:
            pred = pred + offset
        residuals = y - pred
        return 0.5 * jnp.sum(residuals**2)

    def _compute_cov_matrix(
        self, X: jnp.ndarray, params: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute covariance matrix for parameters with offset."""
        n = X.shape[0]
        pred = X @ params
        if offset is not None:
            pred = pred + offset
        residuals = y - pred
        sigma2 = jnp.sum(residuals**2) / (n - len(params))
        return sigma2 * jnp.linalg.pinv(X.T @ X)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None) -> dict:
        """Fit linear regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        offset : jnp.ndarray | None, default=None
            Offset term to include in the model. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        Dictionary containing:

                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
        """
        # Fit model
        params = self._exact_solution(X, y, offset)

        # Compute standard errors
        llf = self.get_llf(X, y, params, offset)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            cov = self._compute_cov_matrix(X, params, y, offset)
            se = jnp.sqrt(jnp.diag(cov))
            stat = (params[-1] / se[-1]) ** 2
            pval = jsp.stats.chi2.sf(stat, df=1)

        return {"coef": params, "llf": llf, "se": se, "stat": stat, "pval": pval}

    def predict(self, X: jnp.ndarray, params: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Predict response variable using fitted model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        offset : jnp.ndarray | None, default=None
            Offset term to include in the prediction. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        jnp.ndarray
            Predicted response variable.
        """
        pred = X @ params
        if offset is not None:
            pred += offset
        return pred


@dataclass(frozen=True)
class LogisticRegression(Regression):
    """Logistic regression in JAX.

    This class implements logistic regression for binary classification tasks
    with support for offset terms. Offsets are added to the linear predictor
    before applying the logistic function.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options are "BFGS" or "IRLS" (recommended).
    skip_stats : bool, default=False
        Whether to skip calculating test statistics.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import LogisticRegression
    >>> X = jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])  # Design matrix with intercept
    >>> y = jnp.array([0.0, 0.0, 1.0])  # Binary outcome
    >>> model = LogisticRegression(optimizer="IRLS")
    >>> result = model.fit(X, y)
    >>> print(f"Coefficients: {result['coef']}")
    """

    def _negative_log_likelihood(
        self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> float:
        """Compute negative log likelihood with offset."""
        logits = jnp.dot(X, params)
        if offset is not None:
            logits = logits + offset
        nll = -jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
        return nll

    def _weight_fn(self, X: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute weights for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        p = jax.nn.sigmoid(eta)
        return p * (1 - p)

    def _working_resid_fn(
        self, X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute working residuals for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        p = jax.nn.sigmoid(eta)
        return eta + (y - p) / jnp.clip(p * (1 - p), 1e-6)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        test_idx: int = -1,
    ) -> dict:
        """Fit logistic regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Binary response vector of shape (n_samples,).
        offset : jnp.ndarray | None, default=None
            Offset term to include in the model. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        Dictionary containing:

                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
        """
        # Fit model
        init_params = jnp.zeros(X.shape[1])
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset)
            params = self._fit_bfgs(nll, init_params)
        elif self.optimizer == "IRLS":
            params = self._fit_irls(X, y, self._weight_fn, self._working_resid_fn, init_params, offset=offset)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, offset)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset)
            se, stat, pval = self._compute_stats(X, nll, params, test_idx=test_idx)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
        }

    def predict(self, X: jnp.ndarray, params: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Predict probabilities using fitted model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        params : jnp.ndarray
            Fitted parameter estimates.
        offset : jnp.ndarray | None, default=None
            Offset term to include in the prediction. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        jnp.ndarray
            Predicted probabilities of the positive class.
        """
        logits = X @ params
        if offset is not None:
            logits += offset
        return jax.nn.sigmoid(logits)


@dataclass(frozen=True)
class NegativeBinomialRegression(Regression):
    """Negative Binomial regression in JAX.

    This class implements Negative Binomial regression for modeling count data,
    particularly RNA-seq data, with support for offsets to incorporate size factors
    or other normalization terms. The model uses a log link function and allows for
    overdispersion in count data.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options are "BFGS" or "IRLS".
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics.
    dispersion : float | None, default=None
        Fixed dispersion parameter. If :obj:`None`, dispersion is estimated from the data.
    dispersion_range : tuple[float, float], default=(1e-6, 10.0)
        Range for the dispersion parameter. Used to constrain the estimated dispersion
        to avoid numerical issues.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import NegativeBinomialRegression
    >>> X = jnp.array([[1.0, 0.0], [1.0, 1.0]])  # Design matrix with intercept
    >>> y = jnp.array([10.0, 20.0])  # Count data
    >>> size_factors = jnp.array([0.8, 1.2])  # Size factors from normalization
    >>> offset = jnp.log(size_factors)  # Log transform for offset
    >>> model = NegativeBinomialRegression(optimizer="IRLS")
    >>> result = model.fit(X, y, offset=offset)
    >>> print(f"Coefficients: {result['coef']}")
    """

    dispersion: float | None = None
    dispersion_range: tuple[float, float] = (1e-8, 100.0)

    def _negative_log_likelihood(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> float:
        """Compute negative log likelihood with offset."""
        eta = X @ params

        if offset is not None:
            eta = eta + offset

        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # TODO: Potentially compute with nb_nll from here
        r = 1 / jnp.clip(dispersion, self.dispersion_range[0], self.dispersion_range[1])

        ll = (
            jsp.special.gammaln(r + y)
            - jsp.special.gammaln(r)
            - jsp.special.gammaln(y + 1)
            + r * jnp.log(r / (r + mu))
            + y * jnp.log(mu / (r + mu))
        )
        return -jnp.sum(ll)

    def _weight_fn(
        self, X: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None, dispersion: float = 1.0
    ) -> jnp.ndarray:
        """Compute weights for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # Negative binomial variance = μ + φμ²
        var = mu + dispersion * mu**2
        # IRLS weights: (dμ/dη)² / var
        # For log link: dμ/dη = μ
        return mu**2 / jnp.clip(var, 1e-6)

    def _working_resid_fn(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        beta: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> jnp.ndarray:
        """Compute working residuals for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # Working response: z = η + (y - μ) * (dη/dμ)
        # For log link: dη/dμ = 1/μ
        return eta + (y - mu) / mu

    def get_llf(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> float:
        """Get log-likelihood at fitted parameters with offset."""
        nll = self._negative_log_likelihood(params, X, y, offset, dispersion)
        return -nll

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        test_idx: int = -1,
    ) -> dict:
        """Fit negative binomial regression model with optional offset.

        This method fits a Negative Binomial regression model to count data,
        with support for including offset terms (typically log size factors)
        to account for normalization. The method also handles dispersion
        estimation if not provided during initialization.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Count response vector of shape (n_samples,).
        offset : jnp.ndarray | None, default=None
            Offset term (log scale) to include in the model. Typically
            log(size_factors) for RNA-seq data. If provided, overrides
            the offset set during class initialization.
        test_idx : int, default=-1
            Index of the parameter to test. If -1, tests the last parameter.

        Returns
        -------
        Dictionary containing:

                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
                - dispersion: Estimated or provided dispersion parameter
        """
        # Estimate dispersion parameter
        if self.dispersion is not None:
            dispersion = jnp.clip(self.dispersion, self.dispersion_range[0], self.dispersion_range[1])
        else:
            dispersion = DispersionEstimator(
                design_matrix=X,
                size_factors=offset if offset is not None else jnp.ones(X.shape[0]),
                min_disp=self.dispersion_range[0],
                max_disp=max(self.dispersion_range[1], X.shape[0]),
            ).fit_dispersion_single_gene(
                counts=y,
            )

        # Initialize parameters
        init_params = jnp.zeros(X.shape[1])

        # Better initialization for intercept
        mean_y = jnp.maximum(jnp.mean(y), 1e-8)
        if offset is not None:
            init_params = init_params.at[0].set(jnp.log(mean_y) - jnp.mean(offset))
        else:
            init_params = init_params.at[0].set(jnp.log(mean_y))

        # Fit model
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset, dispersion=dispersion)
            params = self._fit_bfgs(nll, init_params)
        elif self.optimizer == "IRLS":
            params = self._fit_irls(
                X, y, self._weight_fn, self._working_resid_fn, init_params, offset=offset, dispersion=dispersion
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, offset, dispersion)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset, dispersion=dispersion)
            se, stat, pval = self._compute_stats(X, nll, params, test_idx=test_idx)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
            "dispersion": dispersion,
        }

    def predict(
        self,
        X: jnp.ndarray,
        params: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Predict count response variable using fitted model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        params : jnp.ndarray
            Fitted parameter estimates.
        offset : jnp.ndarray | None, default=None
            Offset term to include in the prediction. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        jnp.ndarray
            Predicted count response variable.
        """
        eta = X @ params
        if offset is not None:
            eta += offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)
        return mu


@dataclass(frozen=True)
class DispersionEstimator:
    """Dispersion estimator in JAX.

    This implementation cosely follows the PyDESeq2 approach:

        - Method of moments and rough dispersion initialization
        - MLE for initial genewise dispersion estimation
        - Iterative trend fitting with outlier filtering
        - MAP estimation for final dispersion values

    Parameters
    ----------
    design_matrix : jnp.ndarray
        Design matrix for the experiment, shape (n_samples, n_covariates).
    size_factors : jnp.ndarray
        Size factors for normalization, shape (n_samples,).
    min_disp : float, default=1e-8
        Minimum allowed dispersion value.
    max_disp : float, default=100.0
        Maximum allowed dispersion value.
    min_mu : float, default=0.5
        Threshold for mean estimates.
    """

    design_matrix: jnp.ndarray = field(compare=False)
    size_factors: jnp.ndarray = field(compare=False)
    min_disp: float = 1e-8
    max_disp: float = 100.0
    min_mu: float = 0.5

    def __post_init__(self):
        """Validate input parameters."""
        if self.design_matrix.ndim != 2:
            raise ValueError("Design matrix must be 2D (n_samples, n_covariates).")
        if self.size_factors.ndim != 1 or self.size_factors.shape[0] != self.design_matrix.shape[0]:
            raise ValueError("Size factors must be 1D and match the number of samples in the design matrix.")
        if not (self.min_disp > 0 and self.max_disp > self.min_disp):
            raise ValueError("Invalid dispersion range: min_disp must be > 0 and max_disp must be > min_disp.")

    @partial(jax.jit, static_argnums=(0,))
    def fit_rough_dispersions_single_gene(self, normed_counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate rough dispersions using linear model residuals (JIT-compiled).

        Parameters
        ----------
        normed_counts : jnp.ndarray
            Normalized count data for a single gene, shape (n_samples,).

        Returns
        -------
        jnp.ndarray
            Rough dispersion estimates clipped to valid range.
        """
        num_samples, num_vars = self.design_matrix.shape

        # Fit linear model
        model = LinearRegression(skip_stats=True)
        results = model.fit(self.design_matrix, normed_counts)

        y_hat = (results["coef"] @ self.design_matrix.T).T  # (num_samples, )
        y_hat = jnp.maximum(y_hat, 1.0)  # Threshold as in PyDESeq2

        nominator = (normed_counts - y_hat) ** 2 - y_hat
        denom = (num_samples - num_vars) * y_hat**2

        dispersions = (nominator / denom).sum(0)

        return jnp.maximum(dispersions, 0)  # Clip negatives as in PyDESeq2

    def fit_rough_dispersions(self, normed_counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate rough dispersions for multiple genes.

        Parameters
        ----------
        normed_counts : jnp.ndarray
            Normalized count data, shape (n_samples, n_genes).

        Returns
        -------
        jnp.ndarray
            Rough dispersion estimates for all genes, shape (n_genes,).
        """
        return jax.vmap(
            self.fit_rough_dispersions_single_gene,
            in_axes=(1,),
        )(normed_counts)

    @partial(jax.jit, static_argnums=(0,))
    def fit_moments_dispersions(self, normed_counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate dispersions using method of moments (JIT-compiled).

        Parameters
        ----------
        normed_counts : jnp.ndarray
            Normalized count data, shape (n_samples, n_genes).

        Returns
        -------
        jnp.ndarray
            Method of moments dispersion estimates, shape (n_genes,).
        """
        # Mean inverse size factor
        s_mean_inv = jnp.mean(1.0 / self.size_factors)

        # Gene-wise means and variances
        mu = jnp.mean(normed_counts, axis=0)
        sigma = jnp.var(normed_counts, axis=0, ddof=1)  # Unbiased estimator

        return jnp.nan_to_num((sigma - s_mean_inv * mu) / mu**2)

    def fit_initial_dispersions(self, counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate initial dispersions as minimum of rough and moments estimates (JIT-compiled).

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data, shape (n_samples, n_genes).

        Returns
        -------
        jnp.ndarray
            Initial dispersion estimates, shape (n_genes,).
        """
        normed_counts = counts / self.size_factors[:, None]

        rough_disp = self.fit_rough_dispersions(normed_counts)
        moments_disp = self.fit_moments_dispersions(normed_counts)

        # Take minimum as in PyDESeq2
        init_disp = jnp.minimum(rough_disp, moments_disp)
        return jnp.minimum(jnp.maximum(init_disp, self.min_disp), self.max_disp)

    def fit_mu_single_gene(self, counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate gene-wise means of the NB distribution (mu).

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data for a single gene, shape (n_samples,).

        Returns
        -------
        jnp.ndarray
            Estimated mean expression values, shape (n_samples,).
        """
        # Fit linear model
        model = LinearRegression(skip_stats=True)
        results = model.fit(self.design_matrix, counts / self.size_factors)

        mu_hat = self.size_factors * (results["coef"] @ self.design_matrix.T)

        # Threshold mu_hat as 1/mu_hat will be used later on.
        return jnp.maximum(mu_hat, self.min_mu)

    def fit_mu(self, counts: jnp.ndarray) -> jnp.ndarray:
        """Estimate gene-wise means of the NB distribution (mu).

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data, shape (n_samples, n_genes).

        Returns
        -------
        jnp.ndarray
            Estimated mean expression values, shape (n_samples, n_genes).
        """
        return jax.vmap(
            self.fit_mu_single_gene,
            in_axes=(1,),
            out_axes=1,
        )(counts)

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def fit_dispersion_mle_single_gene(
        self,
        counts: jnp.ndarray,
        mu: jnp.ndarray,
        alpha_init: float,
        prior_disp_var: float = 1.0,
        use_prior_reg: bool = False,
        use_cr_reg: bool = True,
    ) -> tuple[float, bool]:
        """Estimate dispersion using MLE following PyDESeq2 exactly.

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data for a single gene, shape (n_samples,).
        mu : jnp.ndarray
            Estimated mean of the NB distribution, shape (n_samples, n_genes).
        alpha_init : float
            Initial dispersion estimate.
        design_matrix : jnp.ndarray
            Design matrix for the experiment, shape (n_samples, n_covariates).
        prior_disp_var : float, default=1.0
            Prior variance for dispersion regularization.
        use_prior_reg : bool, default=False
            Whether to use prior regularization.
        use_cr_reg : bool, default=True
            Whether to use Cox-Reid regularization.

        Returns
        -------
        tuple[float, bool]
            Tuple containing (optimized_dispersion, optimization_success).
        """
        alpha_init = jnp.clip(alpha_init, self.min_disp, self.max_disp)
        log_alpha_init = jnp.log(alpha_init)
        # Precompute design matrix transpose for efficiency
        XTX = self.design_matrix.T @ self.design_matrix

        def loss(log_alpha: jnp.ndarray) -> jnp.ndarray:
            # closure to be minimized
            alpha = jnp.exp(log_alpha)
            reg = 0

            if use_cr_reg:
                W = mu / (1 + mu * alpha)
                reg += 0.5 * safe_slogdet(XTX * W.sum())[1]

            if use_prior_reg:
                reg += (log_alpha - log_alpha_init) ** 2 / (2 * prior_disp_var)

            return nb_nll(counts, mu, alpha) + reg

        init_params = jnp.array([log_alpha_init])

        res = optimize.minimize(
            lambda x: loss(x[0]),
            x0=init_params,
            method="BFGS",
            # This ensures better convergence similar to PyDESeq2
            options={"maxiter": 100, "gtol": 1e-2},
        )

        # If optimization fails, fallback to grid search
        log_alpha_opt = jax.lax.cond(
            res.success,
            lambda x: x[0],
            lambda _: grid_fit_alpha(
                counts,
                self.design_matrix,
                mu,
                alpha_init,
                self.min_disp,
                self.max_disp,
                prior_disp_var,
                use_prior_reg,
                use_cr_reg,
                grid_length=100,
            ),
            res.x,
        )

        log_alpha_opt = jnp.clip(jnp.exp(res.x[0]), self.min_disp, self.max_disp)
        return log_alpha_opt, res.success

    def fit_dispersion_mle(
        self,
        counts: jnp.ndarray,
        mu: jnp.ndarray,
        alpha_init: jnp.ndarray,
        prior_disp_var: float = 1.0,
        use_prior_reg: bool = False,
        use_cr_reg: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Estimate gene-wise dispersion using MLE.

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data, shape (n_samples, n_genes).
        mu : jnp.ndarray
            Estimated mean of the NB distribution, shape (n_samples, n_genes).
        alpha_init : jnp.ndarray
            Initial dispersion estimates, shape (n_genes,).
        prior_disp_var : float, default=1.0
            Prior variance for dispersion regularization.
        use_prior_reg : bool, default=False
            Whether to use prior regularization.
        use_cr_reg : bool, default=True
            Whether to use Cox-Reid regularization.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Tuple containing (optimized_dispersions, optimization_success_flags).
            optimized_dispersions: Estimated dispersion values, shape (n_genes,).
            optimization_success_flags: Boolean flags indicating success for each gene, shape (n_genes,).
        """

        def fit_dispersion(x, m, a):
            return self.fit_dispersion_mle_single_gene(
                x,
                m,
                a,
                prior_disp_var,
                use_prior_reg,
                use_cr_reg,
            )

        return jax.vmap(
            fit_dispersion,
            in_axes=(1, 1, 0),
        )(counts, mu, alpha_init)

    def fit_dispersion_single_gene(
        self,
        counts: jnp.ndarray,
    ) -> tuple[float, bool]:
        """Estimate gene-wise dispersion using initial dispersions and MLE.

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data for a single gene, shape (n_samples,).

        Returns
        -------
        tuple[float, bool]
            Tuple containing (estimated_dispersion, success_flag).
            estimated_dispersion: Estimated dispersion value for the gene.
            success_flag: Boolean indicating if the optimization was successful.
        """
        # Estimate initial dispersion
        normed_counts = counts / self.size_factors
        initial_dispersions = self.fit_initial_dispersions(normed_counts[:, None])
        mu = self.fit_mu_single_gene(counts)

        # Fit MLE dispersion
        alpha_init = initial_dispersions[0]
        dispersion, _ = self.fit_dispersion_mle_single_gene(
            counts,
            mu,
            alpha_init,
            prior_disp_var=1.0,
            use_prior_reg=False,
            use_cr_reg=True,
        )

        return dispersion

    def fit_mean_dispersion_trend(self, dispersions: jnp.ndarray) -> jnp.ndarray:
        """Fit mean trend (constant dispersion).

        Parameters
        ----------
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates, shape (n_genes,).

        Returns
        -------
        jnp.ndarray
            Constant trend fitted to dispersions, shape (n_genes,).
        """
        mean_disp = trim_mean(
            dispersions[dispersions > 10 * self.min_disp],
            proportiontocut=0.001,
        )
        return np.full_like(dispersions, mean_disp)

    def dispersion_trend_gamma_glm(
        self, covariates: jnp.ndarray, dispersions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        """Fit dispersion trend using Gamma GLM.

        Parameters
        ----------
        covariates : jnp.ndarray
            Covariate values (1/mu), shape (n_genes,).
        dispersions : jnp.ndarray
            Dispersion estimates, shape (n_genes,).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, bool]
            Tuple containing (coefficients, predictions, success_flag).
        """
        # Add intercept column: [1, 1/μ]
        n_samples = covariates.shape[0]
        covariates_w_intercept = np.column_stack([np.ones(n_samples), covariates])

        def loss(coeffs):
            mu = covariates_w_intercept @ coeffs
            return np.nanmean(dispersions / mu + np.log(mu), axis=0)

        def grad(coeffs):
            mu = covariates_w_intercept @ coeffs
            return -np.nanmean(((dispersions / mu - 1)[:, None] * covariates_w_intercept) / mu[:, None], axis=0)

        try:
            res = scipy_optimize.minimize(
                loss,
                x0=np.array([1.0, 1.0]),
                jac=grad,
                method="L-BFGS-B",
                bounds=[(1e-12, np.inf)],
            )

        except RuntimeWarning:  # Could happen if the coefficients fall to zero
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), False

        coeffs = res.x
        return coeffs, covariates_w_intercept @ coeffs, res.success

    def fit_parametric_dispersion_trend(
        self,
        dispersions: jnp.ndarray,
        normed_means: jnp.ndarray,
    ) -> jnp.ndarray:
        """Fit parametric dispersion trend: f(μ) = α₁/μ + α₀.

        This method exactly mimics PyDESeq2's parametric trend fitting with
        iterative outlier removal and convergence checking.

        Parameters
        ----------
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates, shape (n_genes,).
        normed_means : jnp.ndarray
            Mean normalized expression values, shape (n_genes,).

        Returns
        -------
        jnp.ndarray
            Fitted parametric trend values, shape (n_genes,).
        """
        covariates = 1.0 / normed_means  # 1/μ

        # Iterative fitting with outlier removal
        old_coeffs = np.array([0.1, 0.1])
        coeffs = np.array([1.0, 1.0])

        current_dispersions = dispersions
        current_covariates = covariates

        # Check convergence
        while (coeffs > 1e-10).all() and (np.log(np.abs(coeffs / old_coeffs)) ** 2).sum() >= 1e-6:
            old_coeffs = coeffs
            # Fit GLM
            coeffs, predictions, glm_success = self.dispersion_trend_gamma_glm(current_covariates, current_dispersions)

            if not glm_success or np.any(coeffs <= 1e-10):
                warnings.warn(
                    "The dispersion trend curve fitting did not converge. Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )
                # Fitting failed, fall back to mean dispersion
                return self.fit_mean_dispersion_trend(dispersions)

            # Filter outliers for next iteration
            pred_ratios = current_dispersions / predictions
            outlier_mask = (pred_ratios < 1e-4) | (pred_ratios >= 15)
            current_dispersions = current_dispersions[~outlier_mask]
            current_covariates = current_covariates[~outlier_mask]

        # Compute final predictions for all genes
        fitted_dispersions = coeffs[0] + coeffs[1] * covariates  # α₀ + α₁/μ

        return fitted_dispersions

    def fit_dispersion_trend(
        self,
        dispersions: jnp.ndarray,
        normed_means: jnp.ndarray,
        trend_type: str = "parametric",
    ) -> jnp.ndarray:
        """Main interface for fitting dispersion trends.

        Parameters
        ----------
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates, shape (n_genes,).
        normed_means : jnp.ndarray
            Mean normalized expression values, shape (n_genes,).
        trend_type : str, default="parametric"
            Type of trend to fit. Options are "parametric" or "mean".

        Returns
        -------
        jnp.ndarray
            Fitted trend values, shape (n_genes,).

        Raises
        ------
        ValueError
            If trend_type is not "parametric" or "mean".
        """
        if trend_type == "parametric":
            return self.fit_parametric_dispersion_trend(dispersions, normed_means)
        elif trend_type == "mean":
            return self.fit_mean_dispersion_trend(dispersions)
        else:
            raise ValueError(f"Unknown trend_type: {trend_type}")

    def fit_dispersion_prior(self, dispersions: jnp.ndarray, trend: jnp.ndarray) -> float:
        """Fit dispersion variance priors and standard deviation of log-residuals.

        Parameters
        ----------
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates, shape (n_genes,).
        trend : jnp.ndarray
            Fitted trend values, shape (n_genes,).

        Returns
        -------
        float
            Prior variance for dispersion estimates.
        """
        # Exclude genes with all zeroes
        num_samples, num_vars = self.design_matrix.shape

        # Check the degrees of freedom
        if (num_samples - num_vars) <= 3:
            warnings.warn(
                "As the residual degrees of freedom is less than 3, the distribution "
                "of log dispersions is especially asymmetric and likely to be poorly "
                "estimated by the MAD.",
                UserWarning,
                stacklevel=2,
            )

        # Fit dispersions to the curve, and compute log residuals
        disp_residuals = np.log(dispersions) - np.log(trend)

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = dispersions >= (100 * self.min_disp)

        squared_logres = mean_absolute_deviation(disp_residuals[above_min_disp]) ** 2

        prior_disp_var = np.maximum(
            squared_logres - polygamma(1, (num_samples - num_vars) / 2),
            0.25,
        )

        return prior_disp_var.item()

    def fit_MAP_dispersions(
        self,
        counts: jnp.ndarray,
        dispersions: jnp.ndarray,
        trend: jnp.ndarray,
        mu: jnp.ndarray,
        prior_disp_var: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.

        Parameters
        ----------
        counts : jnp.ndarray
            Raw count data, shape (n_samples, n_genes).
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates, shape (n_genes,).
        trend : jnp.ndarray
            Fitted trend values, shape (n_genes,).
        mu : jnp.ndarray
            Estimated mean of the NB distribution, shape (n_samples, n_genes).
        prior_disp_var : float | None, default=None
            Prior variance for dispersion estimates. If :obj:`None`, it will be estimated
            from the data using the `fit_dispersion_prior` method.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Tuple containing (MAP dispersion estimates, success flags).
            MAP dispersion estimates: Estimated dispersion values, shape (n_genes,).
            success flags: Boolean flags indicating convergence success for each gene, shape (n_genes,).
        """
        if prior_disp_var is None:
            # Compute prior variance for dispersion
            prior_disp_var = self.fit_dispersion_prior(
                dispersions=dispersions,
                trend=trend,
            )

        return self.fit_dispersion_mle(
            counts=counts,
            mu=mu,
            alpha_init=trend,
            prior_disp_var=prior_disp_var,
            use_prior_reg=True,
            use_cr_reg=True,
        )
