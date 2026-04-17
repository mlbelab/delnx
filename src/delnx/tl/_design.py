"""Design matrix construction from formulas or condition keys.

Shared helper used by both :func:`nb_fit` and :func:`de` to build design
matrices with consistent encoding.
"""

import numpy as np
import pandas as pd
import patsy


def build_design(
    obs: pd.DataFrame,
    formula: str | None = None,
    condition_key: str | None = None,
    reference: str | None = None,
    covariate_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a design matrix from a formula or condition_key.

    Parameters
    ----------
    obs : pd.DataFrame
        Observation metadata (typically ``adata.obs``).
    formula : str | None
        R-style formula (e.g., ``"~ treatment + batch"``).
        Mutually exclusive with ``condition_key``.
    condition_key : str | None
        Column in ``obs`` for condition labels. Builds a formula internally.
        Mutually exclusive with ``formula``.
    reference : str | None
        Reference level for the condition. Used to reorder categorical levels
        so the reference becomes the intercept. Only used with ``condition_key``
        or when ``formula`` contains a simple term matching a categorical column.
    covariate_keys : list[str] | None
        Additional columns to include as covariates. Only used with
        ``condition_key`` (ignored when ``formula`` is provided, since
        covariates should be part of the formula).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Design matrix of shape ``(n_obs, n_coef)`` and list of column names.

    Raises
    ------
    ValueError
        If both ``formula`` and ``condition_key`` are specified, or neither is.
    """
    if formula is not None and condition_key is not None:
        raise ValueError("Specify either 'formula' or 'condition_key', not both.")
    if formula is None and condition_key is None:
        raise ValueError("One of 'formula' or 'condition_key' must be specified.")

    if condition_key is not None:
        # Build formula from condition_key + covariates
        if condition_key not in obs.columns:
            raise ValueError(f"Condition key '{condition_key}' not found in obs")
        formula = f"~ {condition_key}"
        if covariate_keys:
            for cov in covariate_keys:
                if cov not in obs.columns:
                    raise ValueError(f"Covariate '{cov}' not found in obs")
            formula += " + " + " + ".join(covariate_keys)

    # Prepare obs copy with reference level handling
    obs_copy = obs.copy()
    if reference is not None and condition_key is not None:
        col = obs_copy[condition_key]
        unique_vals = col.unique() if not hasattr(col, "cat") else col.cat.categories.tolist()
        if reference not in list(unique_vals):
            raise ValueError(
                f"Reference '{reference}' not found in '{condition_key}'. "
                f"Available levels: {list(unique_vals)}"
            )
        # Reorder categories so reference is first (becomes intercept in Treatment coding)
        ordered = [reference] + [v for v in unique_vals if v != reference]
        obs_copy[condition_key] = pd.Categorical(col, categories=ordered)

    # Build design matrix via patsy
    dm = patsy.dmatrix(formula, obs_copy, return_type="dataframe")
    design_matrix = np.asarray(dm, dtype=np.float64)
    column_names = list(dm.columns)

    return design_matrix, column_names
