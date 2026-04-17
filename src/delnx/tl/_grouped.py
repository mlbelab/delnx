"""Grouped differential expression wrapper.

Runs any DE function per group (e.g., cell type) and combines results
with cross-group multiple testing correction.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData

from delnx._logging import logger


def grouped(
    func: Callable[..., pd.DataFrame],
    adata: AnnData,
    group_key: str,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Run a DE function separately for each group and combine results.

    Thin orchestrator that subsets ``adata`` by each unique value of
    ``adata.obs[group_key]``, calls ``func`` on each subset, and
    re-corrects p-values across all groups.

    Parameters
    ----------
    func : callable
        DE function with signature ``func(adata, **kwargs) -> pd.DataFrame``.
        The returned DataFrame must contain a ``pval`` column.
        Works with :func:`de`, :func:`rank_de`, or any custom function
        (e.g., a lambda wrapping :func:`nb_fit` + :func:`nb_test`).
    adata : AnnData
        Annotated data object.
    group_key : str
        Column in ``adata.obs`` defining groups (e.g., ``"cell_type"``).
    min_samples : int, default=2
        Minimum observations per group to run the analysis. Groups with
        fewer observations are skipped with a warning.
    multitest_method : str, default="fdr_bh"
        Method for multiple testing correction across all groups
        (see :func:`statsmodels.stats.multipletests`).
    verbose : bool, default=True
        Whether to print progress messages.
    **kwargs
        Passed through to ``func``.

    Returns
    -------
    pd.DataFrame
        Combined results with an additional ``group`` column. The ``padj``
        column is re-computed across all groups.

    Examples
    --------
    Per-cell-type logistic regression:

    >>> results = dx.tl.grouped(dx.tl.de, adata, group_key="cell_type",
    ...                         condition_key="treatment", reference="control",
    ...                         contrast="treatment[T.drugA]")

    Per-cell-type rank-based markers:

    >>> results = dx.tl.grouped(dx.tl.rank_de, adata, group_key="cell_type", condition_key="treatment")

    Per-cell-type negative binomial DE:

    >>> def nb_de(adata, **kw):
    ...     fit = dx.tl.nb_fit(adata, **kw)
    ...     return dx.tl.nb_test(adata, fit)
    >>> results = dx.tl.grouped(nb_de, adata, group_key="cell_type", condition_key="treatment")
    """
    if group_key not in adata.obs.columns:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")

    results = []
    for group in adata.obs[group_key].unique():
        mask = adata.obs[group_key].values == group
        n_obs = np.sum(mask)

        if n_obs < min_samples:
            logger.warning(f"Skipping group '{group}' with {n_obs} < {min_samples} samples", verbose=verbose)
            continue

        logger.info(f"Running DE for group: {group}", verbose=verbose)

        try:
            group_results = func(adata[mask, :], **kwargs)
            group_results["group"] = group
            results.append(group_results)
        except ValueError as e:
            logger.warning(f"DE failed for group '{group}': {e}. Skipping.", verbose=verbose)
            continue

    if not results:
        raise ValueError(
            "Differential expression analysis failed for all groups. "
            "Check input data or set verbose=True for details."
        )

    results = pd.concat(results, axis=0).reset_index(drop=True)

    if "pval" not in results.columns:
        raise ValueError("DE function must return a DataFrame with a 'pval' column.")

    if results["pval"].notna().any():
        # Re-correct p-values across all groups
        valid = results["pval"].notna()
        padj = sm.stats.multipletests(results.loc[valid, "pval"].values, method=multitest_method)[1]
        results["padj"] = np.nan
        results.loc[valid, "padj"] = padj

    # Sort by group, then by padj
    sort_cols = ["group"]
    if "test_condition" in results.columns:
        sort_cols += ["test_condition", "ref_condition"]
    if "condition" in results.columns:
        sort_cols += ["condition"]
    sort_cols.append("padj")
    # Only sort by columns that exist
    sort_cols = [c for c in sort_cols if c in results.columns]
    results = results.sort_values(by=sort_cols).reset_index(drop=True)

    return results
