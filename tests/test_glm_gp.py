"""Tests for glmGamPoi-style models."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from delnx.models._glm_gp import (
    compute_gp_deviance,
    estimate_dispersion_mle,
    estimate_dispersion_moments,
    fit_beta_fisher_scoring,
    fit_beta_one_group,
    _cox_reid_adjustment,
    _nb_nll_full,
)


class TestDeviance:
    """Tests for deviance computation."""

    def test_deviance_perfect_fit(self):
        """Deviance should be 0 when mu equals counts."""
        counts = jnp.array([1.0, 5.0, 10.0, 20.0])
        mu = counts  # Perfect fit
        dispersion = 0.1

        dev = compute_gp_deviance(counts, mu, dispersion)
        assert dev < 1e-6, f"Deviance for perfect fit should be ~0, got {dev}"

    def test_deviance_positive(self):
        """Deviance should be positive for imperfect fit."""
        counts = jnp.array([1.0, 5.0, 10.0, 20.0])
        mu = jnp.array([2.0, 4.0, 12.0, 18.0])  # Imperfect fit
        dispersion = 0.1

        dev = compute_gp_deviance(counts, mu, dispersion)
        assert dev > 0, f"Deviance should be positive, got {dev}"

    def test_deviance_with_zeros(self):
        """Deviance should handle zero counts."""
        counts = jnp.array([0.0, 0.0, 5.0, 10.0])
        mu = jnp.array([1.0, 2.0, 5.0, 10.0])
        dispersion = 0.5

        dev = compute_gp_deviance(counts, mu, dispersion)
        assert jnp.isfinite(dev), f"Deviance should be finite, got {dev}"
        assert dev > 0, f"Deviance should be positive, got {dev}"


class TestFisherScoring:
    """Tests for Fisher-scoring beta estimation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n_samples = 100

        # Design matrix with intercept
        design = jnp.ones((n_samples, 1))

        # Size factors
        size_factors = np.random.uniform(0.5, 2.0, n_samples)
        offset = jnp.log(jnp.array(size_factors))

        # True parameters
        true_beta = jnp.array([3.0])  # log(mean) ~ 3, so mean ~ 20
        true_dispersion = 0.1

        # Generate counts from negative binomial
        true_mu = size_factors * np.exp(3.0)
        r = 1.0 / true_dispersion
        p = r / (r + true_mu)
        counts = jnp.array(stats.nbinom.rvs(r, p, size=n_samples).astype(float))

        return {
            "counts": counts,
            "design": design,
            "offset": offset,
            "true_beta": true_beta,
            "dispersion": true_dispersion,
        }

    @pytest.fixture
    def two_group_data(self):
        """Create two-group comparison data."""
        np.random.seed(123)
        n_per_group = 50
        n_samples = 2 * n_per_group

        # Design matrix: intercept + group indicator
        design = jnp.array(
            [[1.0, 0.0]] * n_per_group + [[1.0, 1.0]] * n_per_group
        )

        # Size factors
        size_factors = np.random.uniform(0.5, 2.0, n_samples)
        offset = jnp.log(jnp.array(size_factors))

        # True parameters: intercept=3, group_effect=1 (2-fold increase)
        true_beta = jnp.array([3.0, 1.0])
        true_dispersion = 0.2

        # Generate counts
        eta = np.array(design @ true_beta) + np.log(size_factors)
        true_mu = np.exp(eta)
        r = 1.0 / true_dispersion
        p = r / (r + true_mu)
        counts = jnp.array(stats.nbinom.rvs(r, p, size=n_samples).astype(float))

        return {
            "counts": counts,
            "design": design,
            "offset": offset,
            "true_beta": true_beta,
            "dispersion": true_dispersion,
        }

    def test_fisher_scoring_convergence(self, simple_data):
        """Fisher-scoring should converge for simple intercept model."""
        init_beta = jnp.array([0.0])

        beta, deviance, converged = fit_beta_fisher_scoring(
            simple_data["counts"],
            simple_data["design"],
            simple_data["offset"],
            simple_data["dispersion"],
            init_beta,
        )

        assert converged, "Fisher-scoring should converge"
        assert jnp.isfinite(deviance), f"Deviance should be finite, got {deviance}"

        # Check that estimated beta is close to true beta
        assert jnp.abs(beta[0] - simple_data["true_beta"][0]) < 0.5, (
            f"Estimated beta {beta[0]} should be close to true {simple_data['true_beta'][0]}"
        )

    def test_fisher_scoring_two_groups(self, two_group_data):
        """Fisher-scoring should recover two-group effects."""
        init_beta = jnp.array([0.0, 0.0])

        beta, deviance, converged = fit_beta_fisher_scoring(
            two_group_data["counts"],
            two_group_data["design"],
            two_group_data["offset"],
            two_group_data["dispersion"],
            init_beta,
        )

        assert converged, "Fisher-scoring should converge for two groups"
        assert jnp.isfinite(deviance), f"Deviance should be finite, got {deviance}"

        # Intercept should be close to true value
        assert jnp.abs(beta[0] - two_group_data["true_beta"][0]) < 0.5, (
            f"Intercept {beta[0]} should be close to {two_group_data['true_beta'][0]}"
        )

        # Group effect should be close to true value
        assert jnp.abs(beta[1] - two_group_data["true_beta"][1]) < 0.5, (
            f"Group effect {beta[1]} should be close to {two_group_data['true_beta'][1]}"
        )

    def test_fisher_scoring_reduces_deviance(self, simple_data):
        """Fisher-scoring should reduce deviance from initial guess."""
        init_beta = jnp.array([0.0])

        # Initial deviance (bad guess)
        mu_init = jnp.exp(simple_data["design"] @ init_beta + simple_data["offset"])
        dev_init = compute_gp_deviance(simple_data["counts"], mu_init, simple_data["dispersion"])

        # Fitted deviance
        beta, dev_final, _ = fit_beta_fisher_scoring(
            simple_data["counts"],
            simple_data["design"],
            simple_data["offset"],
            simple_data["dispersion"],
            init_beta,
        )

        assert dev_final < dev_init, (
            f"Final deviance {dev_final} should be less than initial {dev_init}"
        )


class TestOneGroup:
    """Tests for one-group (intercept-only) fast path."""

    def test_one_group_convergence(self):
        """One-group estimator should converge."""
        np.random.seed(42)
        n_samples = 100

        size_factors = np.random.uniform(0.5, 2.0, n_samples)
        true_mu = size_factors * 20.0
        dispersion = 0.1
        r = 1.0 / dispersion
        p = r / (r + true_mu)
        counts = jnp.array(stats.nbinom.rvs(r, p, size=n_samples).astype(float))

        beta0, deviance, converged = fit_beta_one_group(
            counts,
            jnp.array(size_factors),
            dispersion,
        )

        assert converged, "One-group estimator should converge"
        assert jnp.isfinite(deviance), f"Deviance should be finite, got {deviance}"

        # exp(beta0) should be close to true mean (20)
        estimated_mean = jnp.exp(beta0)
        assert jnp.abs(estimated_mean - 20.0) < 5.0, (
            f"Estimated mean {estimated_mean} should be close to 20"
        )

    def test_one_group_matches_fisher(self):
        """One-group should give similar results to full Fisher-scoring."""
        np.random.seed(42)
        n_samples = 50

        size_factors = np.random.uniform(0.8, 1.2, n_samples)
        true_mu = size_factors * 15.0
        dispersion = 0.2
        r = 1.0 / dispersion
        p = r / (r + true_mu)
        counts = jnp.array(stats.nbinom.rvs(r, p, size=n_samples).astype(float))

        # One-group path
        beta0_fast, dev_fast, _ = fit_beta_one_group(
            counts,
            jnp.array(size_factors),
            dispersion,
        )

        # Full Fisher-scoring with intercept-only design
        design = jnp.ones((n_samples, 1))
        offset = jnp.log(jnp.array(size_factors))
        beta_full, dev_full, _ = fit_beta_fisher_scoring(
            counts,
            design,
            offset,
            dispersion,
            jnp.array([0.0]),
        )

        # Should give similar beta
        assert jnp.abs(beta0_fast - beta_full[0]) < 0.1, (
            f"One-group beta {beta0_fast} should match Fisher {beta_full[0]}"
        )

        # Should give similar deviance
        assert jnp.abs(dev_fast - dev_full) < 1.0, (
            f"One-group deviance {dev_fast} should match Fisher {dev_full}"
        )


class TestDispersionEstimation:
    """Tests for dispersion estimation."""

    @pytest.fixture
    def dispersion_data(self):
        """Create data for dispersion estimation tests."""
        np.random.seed(42)
        n_samples = 100

        true_dispersion = 0.3
        true_mean = 50.0
        size_factors = np.random.uniform(0.5, 2.0, n_samples)
        true_mu = size_factors * true_mean

        r = 1.0 / true_dispersion
        p = r / (r + true_mu)
        counts = jnp.array(stats.nbinom.rvs(r, p, size=n_samples).astype(float))

        return {
            "counts": counts,
            "mu": jnp.array(true_mu),
            "true_dispersion": true_dispersion,
            "design": jnp.ones((n_samples, 1)),
        }

    def test_moment_estimation(self, dispersion_data):
        """Moment-based dispersion should be reasonable."""
        disp_moments = estimate_dispersion_moments(
            dispersion_data["counts"],
            dispersion_data["mu"],
        )

        assert jnp.isfinite(disp_moments), f"Dispersion should be finite, got {disp_moments}"
        assert disp_moments > 0, f"Dispersion should be positive, got {disp_moments}"

        # Should be in reasonable range of true value
        assert 0.05 < disp_moments < 2.0, (
            f"Dispersion {disp_moments} should be in reasonable range"
        )

    def test_mle_estimation(self, dispersion_data):
        """MLE dispersion should be close to true value."""
        init_disp = estimate_dispersion_moments(
            dispersion_data["counts"],
            dispersion_data["mu"],
        )

        disp_mle, converged = estimate_dispersion_mle(
            dispersion_data["counts"],
            dispersion_data["mu"],
            dispersion_data["design"],
            init_disp,
            do_cox_reid=True,
        )

        assert jnp.isfinite(disp_mle), f"MLE dispersion should be finite, got {disp_mle}"
        assert disp_mle > 0, f"MLE dispersion should be positive, got {disp_mle}"

        # Should be reasonably close to true value
        assert jnp.abs(disp_mle - dispersion_data["true_dispersion"]) < 0.3, (
            f"MLE dispersion {disp_mle} should be close to true {dispersion_data['true_dispersion']}"
        )

    def test_cox_reid_adjustment(self, dispersion_data):
        """Cox-Reid adjustment should be finite."""
        cr_term = _cox_reid_adjustment(
            dispersion_data["design"],
            dispersion_data["mu"],
            dispersion_data["true_dispersion"],
        )

        assert jnp.isfinite(cr_term), f"Cox-Reid term should be finite, got {cr_term}"


class TestNegativeLogLikelihood:
    """Tests for negative log-likelihood computation."""

    def test_nll_positive(self):
        """NLL should be positive for reasonable data."""
        counts = jnp.array([5.0, 10.0, 15.0, 20.0])
        mu = jnp.array([6.0, 9.0, 16.0, 18.0])
        dispersion = 0.1

        nll = _nb_nll_full(counts, mu, dispersion)
        assert jnp.isfinite(nll), f"NLL should be finite, got {nll}"

    def test_nll_with_zeros(self):
        """NLL should handle zero counts."""
        counts = jnp.array([0.0, 0.0, 5.0, 10.0])
        mu = jnp.array([1.0, 2.0, 5.0, 10.0])
        dispersion = 0.5

        nll = _nb_nll_full(counts, mu, dispersion)
        assert jnp.isfinite(nll), f"NLL should be finite with zeros, got {nll}"

    def test_nll_increases_with_worse_fit(self):
        """NLL should increase when mu is further from counts."""
        counts = jnp.array([10.0, 20.0, 30.0, 40.0])
        mu_good = jnp.array([10.0, 20.0, 30.0, 40.0])  # Perfect fit
        mu_bad = jnp.array([5.0, 10.0, 60.0, 80.0])  # Poor fit
        dispersion = 0.1

        nll_good = _nb_nll_full(counts, mu_good, dispersion)
        nll_bad = _nb_nll_full(counts, mu_bad, dispersion)

        assert nll_bad > nll_good, (
            f"NLL for bad fit {nll_bad} should exceed good fit {nll_good}"
        )


class TestGLMGPIntegration:
    """Integration tests for glm_gp and test_de functions."""

    @pytest.fixture
    def simple_adata(self):
        """Create simple AnnData for testing."""
        import anndata

        np.random.seed(42)
        n_samples = 50
        n_genes = 100

        # Two groups
        group = np.array(["A"] * 25 + ["B"] * 25)

        # Generate counts with differential expression
        # Group B has higher expression for first 20 genes
        base_expr = np.random.poisson(20, (n_samples, n_genes)).astype(float)

        # Add differential expression
        de_genes = 20
        fold_change = 2.0
        base_expr[25:, :de_genes] *= fold_change

        adata = anndata.AnnData(X=base_expr)
        adata.obs["group"] = group
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        return adata

    @pytest.fixture
    def overdispersed_adata(self):
        """Create AnnData with overdispersed counts."""
        import anndata

        np.random.seed(123)
        n_samples = 60
        n_genes = 80

        # Three groups
        group = np.array(["ctrl"] * 20 + ["treat_a"] * 20 + ["treat_b"] * 20)

        # Generate NB counts
        base_mean = 50
        dispersion = 0.3

        counts = np.zeros((n_samples, n_genes))
        for i in range(n_genes):
            mu = base_mean * np.random.uniform(0.5, 2.0)
            r = 1.0 / dispersion
            p = r / (r + mu)
            counts[:, i] = stats.nbinom.rvs(r, p, size=n_samples)

        # Add DE for first 15 genes in treat_a
        for i in range(15):
            counts[20:40, i] *= 1.5

        adata = anndata.AnnData(X=counts)
        adata.obs["treatment"] = group
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        return adata

    def test_glm_gp_basic(self, simple_adata):
        """Test basic glm_gp fitting."""
        from delnx.tl._glm_gp import glm_gp

        fit = glm_gp(
            simple_adata,
            condition_key="group",
            verbose=False,
        )

        n_genes = simple_adata.n_vars
        n_samples = simple_adata.n_obs

        # Check output shapes
        assert fit.Beta.shape == (n_genes, 2), f"Beta shape: {fit.Beta.shape}"
        assert fit.overdispersions.shape == (n_genes,)
        assert fit.Mu.shape == (n_samples, n_genes)
        assert fit.size_factors.shape == (n_samples,)
        assert fit.deviances.shape == (n_genes,)

        # Check values are reasonable
        assert np.all(fit.overdispersions >= 0), "Dispersions should be non-negative"
        assert np.all(fit.Mu > 0), "Mu should be positive"
        assert np.all(fit.size_factors > 0), "Size factors should be positive"

    def test_glm_gp_intercept_only(self, simple_adata):
        """Test glm_gp with intercept-only model."""
        from delnx.tl._glm_gp import glm_gp

        fit = glm_gp(
            simple_adata,
            condition_key=None,  # Intercept only
            verbose=False,
        )

        assert fit.Beta.shape[1] == 1, "Should have only intercept"
        assert np.all(np.isfinite(fit.Beta)), "Coefficients should be finite"

    def test_glm_gp_no_shrinkage(self, simple_adata):
        """Test glm_gp without QL shrinkage."""
        from delnx.tl._glm_gp import glm_gp

        fit = glm_gp(
            simple_adata,
            condition_key="group",
            overdispersion_shrinkage=False,
            verbose=False,
        )

        assert fit.ql_dispersions is None, "QL dispersions should be None"
        assert fit.df0_prior == 0.0, "Prior df should be 0"

    def test_glm_gp_with_shrinkage(self, overdispersed_adata):
        """Test glm_gp with QL shrinkage."""
        from delnx.tl._glm_gp import glm_gp

        fit = glm_gp(
            overdispersed_adata,
            condition_key="treatment",
            overdispersion_shrinkage=True,
            verbose=False,
        )

        n_genes = overdispersed_adata.n_vars

        assert fit.ql_dispersions is not None, "Should have QL dispersions"
        assert fit.ql_dispersions.shape == (n_genes,)
        assert fit.dispersion_trend is not None, "Should have dispersion trend"
        assert fit.df0_prior > 0, "Prior df should be positive"

    def test_test_de_basic(self, simple_adata):
        """Test basic differential expression testing."""
        from delnx.tl._glm_gp import glm_gp, test_de

        fit = glm_gp(
            simple_adata,
            condition_key="group",
            verbose=False,
        )

        results = test_de(fit)

        # Check output structure
        assert "feature" in results.columns
        assert "log2fc" in results.columns
        assert "pval" in results.columns
        assert "padj" in results.columns

        # Check values
        assert len(results) == simple_adata.n_vars
        assert np.all(results["pval"] >= 0) and np.all(results["pval"] <= 1)
        assert np.all(results["padj"] >= 0) and np.all(results["padj"] <= 1)

    def test_test_de_detects_de_genes(self, simple_adata):
        """Test that DE genes are detected."""
        from delnx.tl._glm_gp import glm_gp, test_de

        fit = glm_gp(
            simple_adata,
            condition_key="group",
            verbose=False,
        )

        results = test_de(fit)

        # First 20 genes should be DE (we added 2-fold change)
        de_genes = [f"gene_{i}" for i in range(20)]
        de_results = results[results["feature"].isin(de_genes)]

        # Most DE genes should have low p-values
        n_significant = (de_results["pval"] < 0.05).sum()
        assert n_significant > 10, f"Should detect most DE genes, got {n_significant}/20"

        # DE genes should have positive log2fc (group B higher)
        mean_lfc = de_results["log2fc"].mean()
        assert mean_lfc > 0.5, f"DE genes should have positive log2fc, got {mean_lfc}"

    def test_test_de_with_ql(self, overdispersed_adata):
        """Test DE testing with QL shrinkage."""
        from delnx.tl._glm_gp import glm_gp, test_de

        fit = glm_gp(
            overdispersed_adata,
            condition_key="treatment",
            overdispersion_shrinkage=True,
            verbose=False,
        )

        results = test_de(fit, contrast=1)  # Test first treatment vs control

        assert len(results) == overdispersed_adata.n_vars
        assert np.all(np.isfinite(results["pval"]))

        # First 15 genes should show DE
        de_genes = [f"gene_{i}" for i in range(15)]
        de_results = results[results["feature"].isin(de_genes)]
        n_sig = (de_results["padj"] < 0.1).sum()
        assert n_sig > 5, f"Should detect DE genes, got {n_sig}/15"


class TestQuasiLikelihood:
    """Tests for quasi-likelihood framework."""

    def test_ql_dispersion_transformation(self):
        """QL dispersions should be close to 1 when MLE matches trend."""
        from delnx.models._quasi_likelihood import compute_ql_dispersions

        n_genes = 100
        np.random.seed(42)

        mu_means = np.random.uniform(10, 100, n_genes)
        dispersions_mle = np.random.uniform(0.1, 0.5, n_genes)
        dispersion_trend = dispersions_mle.copy()  # Perfect trend match

        ql_disp = compute_ql_dispersions(dispersions_mle, mu_means, dispersion_trend)

        # Should be close to 1 when MLE matches trend
        assert np.allclose(ql_disp, 1.0, atol=1e-6), (
            f"QL dispersions should be ~1 when MLE matches trend"
        )

    def test_ql_dispersion_overdispersed(self):
        """QL dispersions should be > 1 for overdispersed genes."""
        from delnx.models._quasi_likelihood import compute_ql_dispersions

        n_genes = 100
        np.random.seed(42)

        mu_means = np.random.uniform(10, 100, n_genes)
        dispersions_mle = np.random.uniform(0.3, 0.6, n_genes)
        dispersion_trend = np.full(n_genes, 0.1)  # Trend is lower

        ql_disp = compute_ql_dispersions(dispersions_mle, mu_means, dispersion_trend)

        # Should be > 1 for overdispersed genes
        assert (ql_disp > 1.0).all(), "QL dispersions should be > 1 for overdispersed genes"

    def test_dispersion_trend_fitting(self):
        """Trend fitting should produce reasonable values."""
        from delnx.models._quasi_likelihood import fit_dispersion_trend

        n_genes = 500
        np.random.seed(42)

        # Create synthetic mean-dispersion relationship
        mean_expression = np.random.uniform(1, 1000, n_genes)
        # Typical pattern: higher dispersion at low expression
        true_trend = 0.1 + 1.0 / np.sqrt(mean_expression)
        dispersions = true_trend * np.random.lognormal(0, 0.3, n_genes)

        fitted_trend = fit_dispersion_trend(dispersions, mean_expression, method="local_median")

        assert len(fitted_trend) == n_genes, "Trend should have same length as input"
        assert np.all(np.isfinite(fitted_trend)), "Trend should be finite"
        assert np.all(fitted_trend > 0), "Trend should be positive"

        # Correlation between fitted and true should be positive
        corr = np.corrcoef(fitted_trend, true_trend)[0, 1]
        assert corr > 0.5, f"Fitted trend should correlate with true trend, got {corr}"

    def test_prior_df_estimation(self):
        """Prior df estimation should produce reasonable values."""
        from delnx.models._quasi_likelihood import _estimate_prior_df

        n_genes = 1000
        np.random.seed(42)

        # Simulate QL dispersions as s2/var0 ~ F(df_residual, df0)
        # so s2 = var0 * F(df_residual, df0)
        true_df0 = 20
        true_s0_sq = 1.0
        df_residual = 10
        ql_dispersions = true_s0_sq * stats.f.rvs(df_residual, true_df0, size=n_genes)

        df0_est, s0_sq_est = _estimate_prior_df(ql_dispersions, df_residual=df_residual)

        assert df0_est > 0, f"Estimated df0 should be positive, got {df0_est}"
        assert s0_sq_est > 0, f"Estimated s0_sq should be positive, got {s0_sq_est}"

        # Estimates should be in reasonable range
        assert 1 < df0_est < 200, f"df0 estimate {df0_est} seems unreasonable"

    def test_ql_shrinkage(self):
        """QL shrinkage should reduce variance."""
        from delnx.models._quasi_likelihood import shrink_ql_dispersions

        n_genes = 500
        np.random.seed(42)

        # Create noisy QL dispersions
        true_disp = 1.0
        ql_dispersions = true_disp + np.random.normal(0, 0.5, n_genes)
        ql_dispersions = np.maximum(ql_dispersions, 0.1)

        shrunken, df0, s0_sq = shrink_ql_dispersions(ql_dispersions, df_residual=10)

        # Variance should be reduced
        var_original = np.var(ql_dispersions)
        var_shrunken = np.var(shrunken)

        assert var_shrunken < var_original, (
            f"Shrinkage should reduce variance: {var_shrunken} >= {var_original}"
        )

        # Should still be positive
        assert np.all(shrunken > 0), "Shrunken dispersions should be positive"

    def test_ql_f_test(self):
        """QL F-test should produce valid p-values."""
        from delnx.models._quasi_likelihood import ql_f_test

        n_genes = 100
        np.random.seed(42)

        # Create test data
        deviance_reduced = np.random.uniform(100, 200, n_genes)
        deviance_full = deviance_reduced - np.random.uniform(0, 20, n_genes)  # Full model better
        ql_dispersions = np.random.uniform(0.5, 1.5, n_genes)

        f_stats, pvals = ql_f_test(
            deviance_full,
            deviance_reduced,
            df_full=50,
            df_reduced=51,
            ql_dispersions=ql_dispersions,
            df0_prior=10.0,
        )

        # F-statistics should be non-negative
        assert np.all(f_stats >= 0), "F-statistics should be non-negative"

        # P-values should be in [0, 1]
        assert np.all(pvals >= 0) and np.all(pvals <= 1), "P-values should be in [0, 1]"

        # Most should be significant (since we created a difference)
        assert np.sum(pvals < 0.05) > n_genes * 0.5, "Most tests should be significant"

    def test_ql_f_test_null(self):
        """QL F-test should give uniform p-values under null."""
        from delnx.models._quasi_likelihood import ql_f_test

        n_genes = 500
        np.random.seed(42)

        # Under null: same deviance for both models
        deviance = np.random.uniform(100, 200, n_genes)
        deviance_full = deviance
        deviance_reduced = deviance + np.random.uniform(0, 0.1, n_genes)  # Tiny difference
        ql_dispersions = np.ones(n_genes)

        f_stats, pvals = ql_f_test(
            deviance_full,
            deviance_reduced,
            df_full=50,
            df_reduced=51,
            ql_dispersions=ql_dispersions,
            df0_prior=10.0,
        )

        # P-values should be roughly uniform
        # Less than 10% should be < 0.05 under null (with some tolerance)
        prop_sig = np.mean(pvals < 0.05)
        assert prop_sig < 0.15, f"Too many significant under null: {prop_sig}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
