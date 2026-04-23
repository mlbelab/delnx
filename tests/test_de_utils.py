"""Tests for utility functions in the DE module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data():
    # Create example data for each type
    rng = np.random.RandomState(0)
    counts = rng.negative_binomial(n=20, p=0.3, size=(10, 5))
    lognorm = np.log1p(counts / counts.sum(axis=0, keepdims=True))
    binary = (counts > counts.mean()).astype(float)
    return {"counts": counts, "lognorm": lognorm, "binary": binary}


def test_infer_data_type(test_data):
    """Test data type inference."""
    from delnx.tl._utils import _infer_data_type

    # Test raw counts detection
    assert _infer_data_type(test_data["counts"]) == "counts"

    # Test log-normalized detection
    assert _infer_data_type(test_data["lognorm"]) == "lognorm"

    # Test binary detection
    assert _infer_data_type(test_data["binary"]) == "binary"


@pytest.mark.parametrize("data_type", ["counts", "lognorm", "binary"])
def test_log2fc(test_data, data_type):
    """Test log2fc calculation for different data types."""
    from delnx.tl._effects import _log2fc

    # Create reference mask (first half ref, second half test)
    ref_mask = np.array([True] * 5 + [False] * 5)

    # Calculate log2fc for each data type
    for _, X in test_data.items():
        log2fc = _log2fc(X, ref_mask, data_type=data_type)

        # Basic checks
        assert isinstance(log2fc, np.ndarray)
        assert log2fc.shape == (X.shape[1],)
        assert not np.any(np.isnan(log2fc))
        assert not np.any(np.isinf(log2fc))

        # Check values match expectations
        eps = 1e-8
        if data_type == "lognorm":
            # For log1p data, first reverse transform then calculate ratios
            ref_counts = np.expm1(X[~ref_mask])
            test_counts = np.expm1(X[ref_mask])
            expected = np.log2((test_counts.mean(axis=0) + eps) / (ref_counts.mean(axis=0) + eps))
        else:
            # For counts and binary, calculate ratio of means directly
            expected = np.log2((X[ref_mask].mean(axis=0) + eps) / (X[~ref_mask].mean(axis=0) + eps))
        np.testing.assert_allclose(log2fc, expected)


def test_invalid_data_type():
    """Test error handling for invalid data type."""
    from delnx.tl._effects import _log2fc

    X = np.random.randn(10, 5)
    ref_mask = np.array([True] * 5 + [False] * 5)

    with pytest.raises(ValueError, match="Unsupported data type"):
        _log2fc(X, ref_mask, data_type="invalid")


@pytest.mark.parametrize(
    "conditions",
    [
        np.array(["A", "A", "B", "B", "C", "C"]),
        pd.Series(["A", "A", "B", "B", "C", "C"]),
        pd.Categorical(["A", "A", "B", "B", "C", "C"]),
    ],
)
def test_validate_conditions(conditions):
    """Test condition validation for different modes."""
    from delnx.tl._utils import _validate_conditions

    # Test all_vs_ref mode
    comps = _validate_conditions(conditions, reference="A", mode="all_vs_ref")
    assert sorted(comps) == sorted([("B", "A"), ("C", "A")])

    # Test all_vs_all mode
    comps = _validate_conditions(conditions, mode="all_vs_all")
    assert sorted(comps) == sorted([("A", "B"), ("A", "C"), ("B", "C")])

    # Test pairwise mode
    comps = _validate_conditions(conditions, reference=("A", "B"), mode="1_vs_1")
    assert comps == [("B", "A")]

    # Test binary conditions
    binary = (conditions == "A").astype(bool)
    comps = _validate_conditions(binary, reference=True, mode="all_vs_ref")
    assert comps == [(False, True)]

    # Test error cases
    with pytest.raises(ValueError, match="Need at least 2 condition levels"):
        _validate_conditions(np.array(["A", "A", "A"]))

    with pytest.raises(ValueError, match="Reference.*not in levels"):
        _validate_conditions(conditions, reference="D", mode="all_vs_ref")

    with pytest.raises(ValueError, match="must be a tuple"):
        _validate_conditions(conditions, reference=None, mode="1_vs_1")


class TestResolveContrast:
    """Tests for resolve_contrast() shorthand resolution."""

    COLUMNS = ["Intercept", "treatment[T.drugA]", "treatment[T.drugB]", "batch[T.b2]"]

    def test_exact_match(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast("treatment[T.drugA]", self.COLUMNS) == 1

    def test_none_returns_last(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast(None, self.COLUMNS) == 3

    def test_int_passthrough(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast(2, self.COLUMNS) == 2

    def test_bracket_shorthand(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast("treatment[drugA]", self.COLUMNS) == 1

    def test_bare_level_with_condition_key(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast("drugA", self.COLUMNS, condition_key="treatment") == 1

    def test_bare_level_suffix_scan(self):
        from delnx.tl._design import resolve_contrast

        assert resolve_contrast("b2", self.COLUMNS) == 3

    def test_ambiguous_bare_level(self):
        from delnx.tl._design import resolve_contrast

        cols = ["Intercept", "x[T.A]", "y[T.A]"]
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_contrast("A", cols)

    def test_not_found(self):
        from delnx.tl._design import resolve_contrast

        with pytest.raises(ValueError, match="not found"):
            resolve_contrast("nonexistent", self.COLUMNS)

    def test_interaction_bracket_shorthand(self):
        from delnx.tl._design import resolve_contrast

        cols = ["Intercept", "a[T.x]", "b[T.y]", "a[T.x]:b[T.y]"]
        assert resolve_contrast("a[x]:b[y]", cols) == 3

    def test_interaction_mixed_shorthand(self):
        from delnx.tl._design import resolve_contrast

        cols = ["Intercept", "a[T.x]", "b[T.y]", "a[T.x]:b[T.y]"]
        assert resolve_contrast("a[x]:b[T.y]", cols) == 3


class TestParseContrastVector:
    """Tests for parse_contrast_vector() formula parsing."""

    COLUMNS = ["Intercept", "treatment[T.drugA]", "treatment[T.drugB]"]

    def test_list_input(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector([0, 1, -1], self.COLUMNS)
        np.testing.assert_array_equal(vec, [0, 1, -1])

    def test_array_input(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector(np.array([0, 1, -1]), self.COLUMNS)
        np.testing.assert_array_equal(vec, [0, 1, -1])

    def test_wrong_length(self):
        from delnx.tl._design import parse_contrast_vector

        with pytest.raises(ValueError, match="length"):
            parse_contrast_vector([0, 1], self.COLUMNS)

    def test_string_formula_subtraction(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector("drugA - drugB", self.COLUMNS, condition_key="treatment")
        np.testing.assert_array_equal(vec, [0, 1, -1])

    def test_string_formula_addition(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector("drugA + drugB", self.COLUMNS, condition_key="treatment")
        np.testing.assert_array_equal(vec, [0, 1, 1])

    def test_string_formula_with_coefficient(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector("0.5*drugA + 0.5*drugB", self.COLUMNS, condition_key="treatment")
        np.testing.assert_array_equal(vec, [0, 0.5, 0.5])

    def test_single_term_returns_none(self):
        from delnx.tl._design import parse_contrast_vector

        assert parse_contrast_vector("drugA", self.COLUMNS) is None

    def test_none_returns_none(self):
        from delnx.tl._design import parse_contrast_vector

        assert parse_contrast_vector(None, self.COLUMNS) is None

    def test_int_returns_none(self):
        from delnx.tl._design import parse_contrast_vector

        assert parse_contrast_vector(1, self.COLUMNS) is None

    def test_bracket_shorthand_in_formula(self):
        from delnx.tl._design import parse_contrast_vector

        vec = parse_contrast_vector("treatment[drugA] - treatment[drugB]", self.COLUMNS)
        np.testing.assert_array_equal(vec, [0, 1, -1])


def test_log2fc_adata(adata_small):
    """Test log2 fold change calculation on AnnData object."""
    import delnx

    # Use the binary layer for testing
    results = delnx.tl.log2fc(adata_small, condition_key="condition")

    assert isinstance(results, pd.DataFrame)
    assert results.shape[0] == adata_small.n_vars
    assert all(col in results.columns for col in ["feature", "log2fc"])
    assert not np.any(np.isnan(results["log2fc"]))
    assert not np.any(np.isinf(results["log2fc"]))
