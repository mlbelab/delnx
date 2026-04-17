"""Regression models in JAX."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import optimize

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
            raise ValueError("A dispersion value must be provided. Use nb_fit() for automatic dispersion estimation.")

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
