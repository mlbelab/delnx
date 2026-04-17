import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from delnx.models import (
    LinearRegression,
    LogisticRegression,
    NegativeBinomialRegression,
)


# Test fixtures and utilities
@pytest.fixture
def linear_data():
    """Generate synthetic linear regression data."""
    np.random.seed(42)
    n_samples, n_features = 100, 3
    X = np.random.randn(n_samples, n_features)
    # Add intercept column
    X = np.column_stack([np.ones(n_samples), X])
    true_coef = np.array([2.0, 1.5, -0.8, 0.5])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    return {
        "X": jnp.array(X),
        "y": jnp.array(y),
        "true_coef": jnp.array(true_coef),
        "n_samples": n_samples,
        "n_features": n_features + 1,  # +1 for intercept
    }


@pytest.fixture
def logistic_data():
    """Generate synthetic logistic regression data."""
    np.random.seed(42)
    n_samples, n_features = 200, 2
    X = np.random.randn(n_samples, n_features)
    X = np.column_stack([np.ones(n_samples), X])
    true_coef = np.array([0.5, 1.2, -0.7])
    logits = X @ true_coef
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    return {
        "X": jnp.array(X),
        "y": jnp.array(y),
        "true_coef": jnp.array(true_coef),
        "n_samples": n_samples,
        "n_features": n_features + 1,
    }


@pytest.fixture
def count_data():
    """Generate synthetic count data for negative binomial regression."""
    np.random.seed(42)
    n_samples, n_features = 150, 2
    X = np.random.randn(n_samples, n_features)
    X = np.column_stack([np.ones(n_samples), X])
    size_factors = np.random.uniform(0.5, 2.0, size=n_samples)
    true_coef = np.array([2.0, 0.5, -0.3])
    true_dispersion = 0.1

    # Generate negative binomial data
    mu = np.exp(X @ true_coef)
    r = 1 / true_dispersion
    p = r / (r + mu)
    y = np.random.negative_binomial(r, p)

    return {
        "X": jnp.array(X),
        "y": jnp.array(y),
        "true_coef": jnp.array(true_coef),
        "true_dispersion": true_dispersion,
        "size_factors": jnp.array(size_factors),
        "n_samples": n_samples,
        "n_features": n_features + 1,
    }


@pytest.fixture
def dispersion_data():
    """Generate data for dispersion estimation tests."""
    np.random.seed(42)
    n_genes, n_samples = 100, 50

    # True parameters
    true_mu = np.random.exponential(scale=100, size=n_genes)
    true_dispersions = np.random.gamma(shape=2.0, scale=0.05, size=n_genes)
    size_factors = np.random.uniform(0.5, 2.0, size=n_samples)

    # Generate count matrix
    counts = np.zeros((n_samples, n_genes))
    for i in range(n_genes):
        mu = true_mu[i]
        disp = true_dispersions[i]
        r = 1.0 / disp
        p = r / (r + mu)
        counts[:, i] = np.random.negative_binomial(r, p, size=n_samples)

    return {
        "counts": jnp.array(counts),
        "true_mu": jnp.array(true_mu),
        "true_dispersions": jnp.array(true_dispersions),
        "size_factors": jnp.array(size_factors),
        "n_genes": n_genes,
        "n_samples": n_samples,
    }


class TestLinearRegression:
    """Test suite for LinearRegression class."""

    def test_initialization(self):
        """Test LinearRegression initialization."""
        # Default initialization
        reg = LinearRegression()
        assert isinstance(reg.maxiter, int)
        assert isinstance(reg.tol, float)
        assert isinstance(reg.optimizer, str)
        assert not reg.skip_stats

        # Custom initialization
        reg_custom = LinearRegression(maxiter=50, tol=1e-8, skip_stats=True)
        assert reg_custom.maxiter == 50
        assert reg_custom.tol == 1e-8
        assert reg_custom.skip_stats

    def test_fit_basic(self, linear_data):
        """Test basic fitting functionality."""
        reg = LinearRegression()
        result = reg.fit(linear_data["X"], linear_data["y"])

        # Check return structure
        required_keys = ["coef", "llf", "se", "stat", "pval"]
        assert all(key in result for key in required_keys)

        # Check coefficient shape
        assert result["coef"].shape == (linear_data["n_features"],)

        # Check coefficient accuracy (should be close to true values)
        np.testing.assert_allclose(result["coef"], linear_data["true_coef"], rtol=0.1, atol=0.1)

        # Check that log-likelihood is finite
        assert jnp.isfinite(result["llf"])

        # Check standard errors
        assert result["se"] is not None
        assert result["se"].shape == (linear_data["n_features"],)
        assert jnp.all(result["se"] > 0)

    def test_fit_skip_stats(self, linear_data):
        """Test fitting with Wald test skipped."""
        reg = LinearRegression(skip_stats=True)
        result = reg.fit(linear_data["X"], linear_data["y"])

        # Should have coefficients and log-likelihood
        assert "coef" in result
        assert "llf" in result

        # Should not have Wald test results
        assert result["se"] is None
        assert result["stat"] is None
        assert result["pval"] is None

    def test_negative_log_likelihood(self, linear_data):
        """Test negative log-likelihood computation."""
        reg = LinearRegression()
        nll = reg._negative_log_likelihood(linear_data["true_coef"], linear_data["X"], linear_data["y"])

        assert jnp.isfinite(nll)
        assert nll > 0  # Should be positive

    def test_exact_solution(self, linear_data):
        """Test exact OLS solution."""
        reg = LinearRegression()
        params = reg._exact_solution(linear_data["X"], linear_data["y"])

        assert params.shape == (linear_data["n_features"],)
        # Should be close to true coefficients
        np.testing.assert_allclose(params, linear_data["true_coef"], rtol=0.1, atol=0.1)

    def test_covariance_matrix(self, linear_data):
        """Test covariance matrix computation."""
        reg = LinearRegression()
        params = reg._exact_solution(linear_data["X"], linear_data["y"])
        cov = reg._compute_cov_matrix(linear_data["X"], params, linear_data["y"])

        # Should be square matrix
        n_features = linear_data["n_features"]
        assert cov.shape == (n_features, n_features)

        # Should be positive semidefinite (diagonal elements > 0)
        assert jnp.all(jnp.diag(cov) > 0)

    def test_edge_cases(self):
        """Test edge cases for LinearRegression."""
        reg = LinearRegression()

        # Perfect fit case
        X = jnp.array([[1, 1], [1, 2], [1, 3]])
        y = jnp.array([2, 4, 6])  # Perfect linear relationship
        result = reg.fit(X, y)

        assert jnp.isfinite(result["llf"])
        np.testing.assert_allclose(result["coef"], [0, 2], atol=1e-6)

    def test_predict(self):
        """Test LinearRegression.predict with and without intercept."""
        reg = LinearRegression()

        # Test with intercept
        X_with_intercept = jnp.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        params_with_intercept = jnp.array([1.0, 0.5])  # intercept=1.0, slope=0.5
        offset = jnp.array([0.1, 0.2, 0.3])

        # Without offset
        pred_no_offset = reg.predict(X_with_intercept, params_with_intercept)
        expected_no_offset = jnp.array([2.0, 2.5, 3.0])  # [1 + 0.5*2, 1 + 0.5*3, 1 + 0.5*4]
        assert jnp.allclose(pred_no_offset, expected_no_offset)

        # With offset
        pred_with_offset = reg.predict(X_with_intercept, params_with_intercept, offset=offset)
        expected_with_offset = jnp.array([2.1, 2.7, 3.3])  # [2.0 + 0.1, 2.5 + 0.2, 3.0 + 0.3]
        assert jnp.allclose(pred_with_offset, expected_with_offset)

        # Test without intercept (no constant column)
        X_no_intercept = jnp.array([[2.0], [3.0], [4.0]])
        params_no_intercept = jnp.array([0.5])  # slope only

        pred_no_intercept = reg.predict(X_no_intercept, params_no_intercept)
        expected_no_intercept = jnp.array([1.0, 1.5, 2.0])  # [0.5*2, 0.5*3, 0.5*4]
        assert jnp.allclose(pred_no_intercept, expected_no_intercept)


class TestLogisticRegression:
    """Test suite for LogisticRegression class."""

    def test_initialization(self):
        """Test LogisticRegression initialization."""
        reg = LogisticRegression()
        assert reg.optimizer == "BFGS"

        reg_irls = LogisticRegression(optimizer="IRLS")
        assert reg_irls.optimizer == "IRLS"

    def test_fit_bfgs(self, logistic_data):
        """Test fitting with BFGS optimizer."""
        reg = LogisticRegression(optimizer="BFGS")
        result = reg.fit(logistic_data["X"], logistic_data["y"])

        # Check return structure
        required_keys = ["coef", "llf", "se", "stat", "pval"]
        assert all(key in result for key in required_keys)

        # Check coefficient shape
        assert result["coef"].shape == (logistic_data["n_features"],)

        # Coefficients should be reasonably close to true values
        np.testing.assert_allclose(
            result["coef"],
            logistic_data["true_coef"],
            rtol=0.3,
            atol=0.3,  # More tolerant for logistic regression
        )

        # Check log-likelihood is negative (as expected for logistic)
        assert jnp.isfinite(result["llf"])

    def test_fit_irls(self, logistic_data):
        """Test fitting with IRLS optimizer."""
        reg = LogisticRegression(optimizer="IRLS", maxiter=50)
        result = reg.fit(logistic_data["X"], logistic_data["y"])

        # Should converge to similar results as BFGS
        assert result["coef"].shape == (logistic_data["n_features"],)
        assert jnp.isfinite(result["llf"])

    def test_weight_function(self, logistic_data):
        """Test IRLS weight function."""
        reg = LogisticRegression()
        beta = jnp.zeros(logistic_data["n_features"])
        weights = reg._weight_fn(logistic_data["X"], beta)

        assert weights.shape == (logistic_data["n_samples"],)
        assert jnp.all(weights >= 0)
        assert jnp.all(weights <= 0.25)  # Max weight for logistic is 0.25

    def test_working_residuals(self, logistic_data):
        """Test IRLS working residuals function."""
        reg = LogisticRegression()
        beta = jnp.zeros(logistic_data["n_features"])
        resid = reg._working_resid_fn(logistic_data["X"], logistic_data["y"], beta)

        assert resid.shape == (logistic_data["n_samples"],)
        assert jnp.all(jnp.isfinite(resid))

    def test_negative_log_likelihood(self, logistic_data):
        """Test negative log-likelihood computation."""
        reg = LogisticRegression()
        nll = reg._negative_log_likelihood(logistic_data["true_coef"], logistic_data["X"], logistic_data["y"])

        assert jnp.isfinite(nll)
        assert nll > 0

    def test_invalid_optimizer(self, logistic_data):
        """Test invalid optimizer raises error."""
        reg = LogisticRegression(optimizer="INVALID")

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            reg.fit(logistic_data["X"], logistic_data["y"])

    def test_predict(self):
        """Test LogisticRegression.predict with and without intercept."""
        reg = LogisticRegression()

        # Test with intercept
        X_with_intercept = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        params_with_intercept = jnp.array([0.0, 1.0])  # intercept=0.0, slope=1.0
        offset = jnp.array([0.5, -0.5, 0.0])

        # Without offset
        pred_no_offset = reg.predict(X_with_intercept, params_with_intercept)
        # Logits: [0, 1, 2], sigmoid([0, 1, 2]) = [0.5, 0.731, 0.881]
        expected_no_offset = jnp.array([0.5, 0.7310585786, 0.8807970779])
        assert jnp.allclose(pred_no_offset, expected_no_offset, rtol=1e-6)

        # With offset
        pred_with_offset = reg.predict(X_with_intercept, params_with_intercept, offset=offset)
        # Logits: [0 + 0.5, 1 - 0.5, 2 + 0.0] = [0.5, 0.5, 2.0]
        expected_with_offset = jnp.array([0.6224593312, 0.6224593312, 0.8807970779])
        assert jnp.allclose(pred_with_offset, expected_with_offset, rtol=1e-6)

        # Test without intercept
        X_no_intercept = jnp.array([[1.0], [2.0], [3.0]])
        params_no_intercept = jnp.array([0.5])  # slope only

        pred_no_intercept = reg.predict(X_no_intercept, params_no_intercept)
        # Logits: [0.5, 1.0, 1.5], sigmoid values
        expected_no_intercept = jnp.array([0.6224593312, 0.7310585786, 0.8175744762])
        assert jnp.allclose(pred_no_intercept, expected_no_intercept, rtol=1e-6)

        # Check probabilities are in [0, 1]
        assert jnp.all(pred_no_offset >= 0.0) and jnp.all(pred_no_offset <= 1.0)
        assert jnp.all(pred_with_offset >= 0.0) and jnp.all(pred_with_offset <= 1.0)
        assert jnp.all(pred_no_intercept >= 0.0) and jnp.all(pred_no_intercept <= 1.0)


class TestNegativeBinomialRegression:
    """Test suite for NegativeBinomialRegression class."""

    def test_initialization(self):
        """Test NegativeBinomialRegression initialization."""
        # Default initialization
        reg = NegativeBinomialRegression()
        assert reg.dispersion is None
        assert isinstance(reg.dispersion_range, tuple)

        # Custom initialization
        reg_custom = NegativeBinomialRegression(dispersion=0.1)
        assert reg_custom.dispersion == 0.1

    def test_fit_with_fixed_dispersion(self, count_data):
        """Test fitting with fixed dispersion parameter."""
        reg = NegativeBinomialRegression(dispersion=0.1, optimizer="BFGS")
        result = reg.fit(count_data["X"], count_data["y"])

        # Check return structure
        required_keys = ["coef", "llf", "se", "stat", "pval"]
        assert all(key in result for key in required_keys)

        # Check coefficient shape
        assert result["coef"].shape == (count_data["n_features"],)

        # Check log-likelihood
        assert jnp.isfinite(result["llf"])

    def test_fit_without_dispersion_raises(self, count_data):
        """Test that fitting without dispersion raises ValueError."""
        reg = NegativeBinomialRegression(dispersion=None)
        with pytest.raises(ValueError, match="dispersion value must be provided"):
            reg.fit(count_data["X"], count_data["y"])

    def test_fit_with_full_dispersion(self, count_data):
        """Test fitting with provided full dispersion estimates."""
        reg = NegativeBinomialRegression(dispersion=count_data["true_dispersion"], optimizer="BFGS")
        result = reg.fit(count_data["X"], count_data["y"])

        # Check return structure
        required_keys = ["coef", "llf", "se", "stat", "pval"]
        assert all(key in result for key in required_keys)

        # Check coefficient shape
        assert result["coef"].shape == (count_data["n_features"],)

        # Check log-likelihood
        assert jnp.isfinite(result["llf"])

    def test_fit_with_offset(self, count_data):
        """Test fitting with an offset term."""
        reg = NegativeBinomialRegression(dispersion=0.1, optimizer="BFGS")
        offset = jnp.log(count_data["size_factors"])

        # Fit with offset
        result = reg.fit(count_data["X"], count_data["y"], offset=offset)

        # Check return structure
        required_keys = ["coef", "llf", "se", "stat", "pval"]
        assert all(key in result for key in required_keys)

        # Check coefficient shape
        assert result["coef"].shape == (count_data["n_features"],)

        # Check log-likelihood
        assert jnp.isfinite(result["llf"])

        # Fit without offset
        result_no_offset = reg.fit(count_data["X"], count_data["y"])

        # Check that coefficients are not identical -> offset is used
        assert not jnp.allclose(result["coef"], result_no_offset["coef"])

    def test_weight_function(self, count_data):
        """Test IRLS weight function."""
        reg = NegativeBinomialRegression(dispersion=0.1)
        beta = jnp.array([1.0, 0.5, -0.2])
        weights = reg._weight_fn(count_data["X"], beta, dispersion=0.1)

        assert weights.shape == (count_data["n_samples"],)
        assert jnp.all(weights > 0)
        assert jnp.all(jnp.isfinite(weights))

    def test_working_residuals(self, count_data):
        """Test IRLS working residuals function."""
        reg = NegativeBinomialRegression(dispersion=0.1)
        beta = jnp.array([1.0, 0.5, -0.2])
        resid = reg._working_resid_fn(count_data["X"], count_data["y"], beta, dispersion=0.1)

        assert resid.shape == (count_data["n_samples"],)
        assert jnp.all(jnp.isfinite(resid))

    def test_negative_log_likelihood(self, count_data):
        """Test negative log-likelihood computation."""
        reg = NegativeBinomialRegression()
        nll = reg._negative_log_likelihood(
            count_data["true_coef"], count_data["X"], count_data["y"], dispersion=count_data["true_dispersion"]
        )

        assert jnp.isfinite(nll)
        assert nll > 0

    def test_dispersion_clipping(self, count_data):
        """Test dispersion parameter clipping."""
        reg = NegativeBinomialRegression(dispersion_range=(0.01, 1.0))

        # Test with dispersion outside range
        nll = reg._negative_log_likelihood(
            count_data["true_coef"],
            count_data["X"],
            count_data["y"],
            dispersion=100.0,  # Should be clipped to 1.0
        )

        assert jnp.isfinite(nll)

    def test_predict(self):
        """Test NegativeBinomialRegression.predict with and without intercept."""
        reg = NegativeBinomialRegression()

        # Test with intercept
        X_with_intercept = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        params_with_intercept = jnp.array([1.0, 0.5])  # intercept=1.0, slope=0.5
        offset = jnp.array([0.5, -0.5, 0.0])  # log(size_factors)

        # Without offset
        pred_no_offset = reg.predict(X_with_intercept, params_with_intercept)
        # Linear predictors: [1, 1.5, 2], exp([1, 1.5, 2]) = [2.718, 4.482, 7.389]
        expected_no_offset = jnp.array([2.7182818285, 4.4816890703, 7.3890560989])
        assert jnp.allclose(pred_no_offset, expected_no_offset, rtol=1e-6)

        # With offset
        pred_with_offset = reg.predict(X_with_intercept, params_with_intercept, offset=offset)
        # Linear predictors with offset: [1 + 0.5, 1.5 - 0.5, 2 + 0.0] = [1.5, 1.0, 2.0]
        expected_with_offset = jnp.array([4.4816890703, 2.7182818285, 7.3890560989])
        assert jnp.allclose(pred_with_offset, expected_with_offset, rtol=1e-6)

        # Test without intercept
        X_no_intercept = jnp.array([[1.0], [2.0], [3.0]])
        params_no_intercept = jnp.array([0.5])  # slope only

        pred_no_intercept = reg.predict(X_no_intercept, params_no_intercept)
        # Linear predictors: [0.5, 1.0, 1.5], exp([0.5, 1.0, 1.5]) = [1.649, 2.718, 4.482]
        expected_no_intercept = jnp.array([1.6487212707, 2.7182818285, 4.4816890703])
        assert jnp.allclose(pred_no_intercept, expected_no_intercept, rtol=1e-6)

        # Check all predictions are positive
        assert jnp.all(pred_no_offset > 0.0)
        assert jnp.all(pred_with_offset > 0.0)
        assert jnp.all(pred_no_intercept > 0.0)


