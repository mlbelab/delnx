import numpy as np
import pandas as pd
from anndata import AnnData

from delnx._constants import COMPATIBLE_DATA_TYPES
from delnx._logging import logger
from delnx._typing import ComparisonMode, DataType
from delnx._utils import _to_dense, _to_list


def _infer_data_type(X: np.ndarray) -> DataType:
    """Infer the type of data from its values.

    Parameters
    ----------
    X
        Expression matrix

    Returns
    -------
    DataType
        Inferred data type:
        - counts: Raw count data (integers, potentially large values)
        - lognorm: Log-normalized data (floating point, typically between 0 and 10)
        - binary: Binary data (only 0s and 1s)
        - scaled: Scaled data (floating point, can be negative or positive)
    """
    # Subsample cells if large
    if X.shape[0] > 300:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], size=300, replace=False)
        sample = X[idx, :]
    else:
        sample = X

    sample = _to_dense(sample).flatten()

    # Check if binary
    unique_vals = np.unique(sample)
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
        return "binary"

    # Check if counts (all non-negative integers)
    is_integer = np.all(np.equal(np.mod(sample, 1), 0))
    is_nonnegative = np.all(sample >= 0)
    if is_integer and is_nonnegative:
        return "counts"

    elif is_nonnegative:
        return "lognorm"

    return "scaled"


def _validate_conditions(
    condition_values: np.ndarray | pd.Series | pd.Categorical,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_ref",
) -> list[tuple[str, str]]:
    """Validate condition values and return valid comparisons.

    Parameters
    ----------
    condition_values
        Array or Series of condition values
    reference
        Reference level for comparisons or tuple of (ref_group, comp_group)
    mode
        How to perform comparisons:
        - all_vs_ref: Compare all levels to reference
        - all_vs_all: Compare all pairs of levels
        - 1_vs_1: Compare only two levels (reference and comparison group)
        - continuous: Compare continuous condition levels (e.g., time points).

    Returns
    -------
    comparisons
        List of tuples with comparisons (level1, level2)
    """
    if mode == "continuous":
        # Check if values are numeric
        if not np.issubdtype(condition_values.dtype, np.number):
            raise ValueError("For continuous mode, condition values must be numeric")
        # Dummhy values to make it compatible with loop
        return [(None, None)]

    # Get unique levels
    levels = sorted(set(_to_list(condition_values)))

    if len(levels) < 2:
        raise ValueError(f"Need at least 2 condition levels, got {len(levels)}: {levels}")

    # Handle different modes
    # Unpack reference if it's a tuple
    ref = None
    alt = None
    if isinstance(reference, tuple):
        ref, alt = reference
    else:
        ref = reference

    if mode == "1_vs_1":
        if not isinstance(reference, tuple):
            raise ValueError("For 1_vs_1 mode, `reference` must be a tuple (ref_group, comp_group)")
        if ref is None or alt is None:
            raise ValueError("For 1_vs_1 mode, both reference and comparison group must be specified")
        if ref not in levels or alt not in levels:
            raise ValueError(f"Reference '{ref}' and comparison group '{alt}' must be in levels: {levels}")
        comparisons = [(alt, ref)]

    elif mode == "all_vs_ref":
        if ref is None:
            raise ValueError("For all_vs_ref mode, reference must be specified")
        elif ref not in levels:
            raise ValueError(f"Reference '{ref}' not in levels: {levels}")
        comparisons = [(level, ref) for level in levels if level != ref]

    elif mode == "all_vs_all":
        comparisons = [(l1, l2) for i, l1 in enumerate(levels) for l2 in levels[i + 1 :]]

    else:
        raise ValueError(f"Invalid comparison mode: {mode}")

    return comparisons


def _prepare_model_data(
    adata: AnnData,
    condition_key: str,
    reference: str,
    mode: ComparisonMode,
    covariate_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare data frame for fitting models."""
    model_data = pd.DataFrame(index=range(adata.n_obs))

    # Set up condition
    if mode == "continuous":
        # For continuous mode, we assume condition_key is numeric and use it as is
        model_data[condition_key] = adata.obs[condition_key].values.astype(float)
    else:
        model_data[condition_key] = (adata.obs[condition_key].values != reference).astype(int)

    # Add covariates
    if covariate_keys is not None:
        for cov in covariate_keys:
            model_data[cov] = adata.obs[cov].values

    return model_data


def _check_method_and_data_type(
    method: str,
    data_type: DataType,
) -> None:
    """Check if the method is compatible with the data type. Raise warnings or errors as appropriate."""
    if method not in COMPATIBLE_DATA_TYPES:
        raise ValueError(f"Method '{method}' is not recognized or supported.")

    if method == "binomial" and data_type not in COMPATIBLE_DATA_TYPES["binomial"]:
        raise ValueError(f"Binomial models require binary data. Current data type is {data_type}.")
    elif method == "lr" and data_type not in COMPATIBLE_DATA_TYPES["lr"]:
        logger.warning(
            f"Logistic regression is designed for {' or '.join(COMPATIBLE_DATA_TYPES['lr'])} data. "
            f"Current data type is {data_type}, which may give unreliable results.",
        )
    elif method.startswith("anova") and data_type not in COMPATIBLE_DATA_TYPES[method]:
        logger.warning(
            f"ANOVA is designed for {' or '.join(COMPATIBLE_DATA_TYPES['anova'])} data. "
            f"Current data type is {data_type}, which may give unreliable results.",
        )
