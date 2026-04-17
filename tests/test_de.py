import numpy as np
import pandas as pd
import pytest

from delnx.tl import de


@pytest.mark.parametrize("method", ["lr", "anova", "anova_residual"])
def test_de_methods_pb_lognorm(adata_pb_lognorm, method):
    """Test different DE methods with lognorm pseudobulk data."""
    results = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
        method=method,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_default_contrast(adata_pb_lognorm):
    """Test that contrast defaults to last coefficient."""
    results = de(
        adata_pb_lognorm,
        condition_key="condition",
        reference="control",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_with_covariates(adata_pb_lognorm):
    """Test DE with condition_key + covariate_keys."""
    adata_pb_lognorm.obs["covariate"] = np.random.rand(adata_pb_lognorm.n_obs)

    results = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
        covariate_keys=["covariate"],
    )
    results_no_cov = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    # Results should differ when covariates are included
    assert not np.allclose(
        results.set_index("feature")["coef"].values,
        results_no_cov.set_index("feature")["coef"].values,
    )


def test_de_formula_basic(adata_pb_lognorm):
    """Test formula-based de()."""
    results = de(
        adata_pb_lognorm,
        formula="~ condition_str",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_formula_with_covariates(adata_pb_lognorm):
    """Test formula with covariates included directly."""
    results = de(
        adata_pb_lognorm,
        formula="~ condition_str + cell_type",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_formula_continuous(adata_pb_lognorm):
    """Test formula with continuous variable."""
    adata_pb_lognorm.obs["score"] = np.random.normal(size=adata_pb_lognorm.n_obs)

    results = de(
        adata_pb_lognorm,
        formula="~ score",
        contrast="score",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


@pytest.mark.parametrize(
    "method,layer",
    [
        ("lr", None),
        ("anova", None),
        ("anova_residual", None),
        ("binomial", "binary"),
    ],
)
def test_de_methods_sc(adata_small, method, layer):
    """Test different DE methods on single-cell data."""
    results = de(
        adata_small,
        condition_key="condition",
        reference="control",
        method=method,
        layer=layer,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_binomial_binary(adata_small):
    """Test binomial DE on binary data."""
    results = de(
        adata_small,
        condition_key="condition",
        reference="control",
        method="binomial",
        layer="binary",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 80
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_errors(adata_pb_lognorm):
    """Test error conditions."""
    # Both formula and condition_key
    with pytest.raises(ValueError, match="either.*formula.*condition_key"):
        de(adata_pb_lognorm, formula="~ condition_str", condition_key="condition_str")

    # Neither formula nor condition_key
    with pytest.raises(ValueError, match="One of.*formula.*condition_key"):
        de(adata_pb_lognorm)

    # Invalid contrast
    with pytest.raises(ValueError, match="not found in design columns"):
        de(adata_pb_lognorm, formula="~ condition_str", contrast="nonexistent")

    # Unsupported method
    with pytest.raises(ValueError, match="Unsupported method"):
        de(adata_pb_lognorm, condition_key="condition_str", reference="control", method="negbinom")

    # Missing condition_key
    with pytest.raises(ValueError, match="not found in obs"):
        de(adata_pb_lognorm, condition_key="nonexistent")


def test_de_data_type_validation(adata_pb_counts):
    """Test data type validation for different DE methods."""
    # Binomial with non-binary data
    with pytest.raises(ValueError, match="require binary data"):
        de(adata_pb_counts, condition_key="condition", reference="control", method="binomial")

    # ANOVA with count data should work (with warning)
    results = de(adata_pb_counts, condition_key="condition", reference="control", method="anova")
    assert len(results) > 0

    # LR with count data should work
    results = de(adata_pb_counts, condition_key="condition", reference="control", method="lr")
    assert len(results) > 0


def test_grouped_de(adata_pb_lognorm):
    """Test grouped DE via public grouped() wrapper."""
    from delnx.tl import grouped

    results = grouped(
        de,
        adata_pb_lognorm,
        group_key="cell_type",
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(col in results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(results["group"].unique()) == 3
    assert "cell_type_1" in results["group"].values.tolist()


def test_grouped_rank_de(adata_pb_lognorm):
    """Test grouped wrapper with rank_de."""
    from delnx.tl import grouped, rank_de

    results = grouped(
        rank_de,
        adata_pb_lognorm,
        group_key="cell_type",
        condition_key="condition_str",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(col in results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(results["group"].unique()) == 3
