"""Multi-group comparison wrapper.

Runs 1-vs-all, pairwise, or all-vs-reference DE comparisons and combines
results with cross-comparison multiple testing correction.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData

from delnx._logging import logger


def compare(
    func: Callable[..., pd.DataFrame],
    adata: AnnData,
    condition_key: str,
    mode: str = "1_vs_all",
    reference: str | None = None,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Run DE comparisons across multiple groups.

    Orchestrator that constructs 1-vs-all, pairwise, or all-vs-reference
    comparisons, calls ``func`` for each, and re-corrects p-values across
    all comparisons.

    Parameters
    ----------
    func : callable
        DE function with signature ``func(adata, **kwargs) -> pd.DataFrame``.
        The returned DataFrame must contain a ``pval`` column.
        Works with :func:`de`, :func:`nb_de`, or any custom function.
    adata : AnnData
        Annotated data object.
    condition_key : str
        Column in ``adata.obs`` defining groups (e.g., ``"cell_type"``).
    mode : str, default="1_vs_all"
        Comparison strategy:

        - ``"1_vs_all"``: Test each group against all others (marker detection).
        - ``"pairwise"``: Test all pairs of groups.
        - ``"all_vs_ref"``: Test each group against a single reference.
    reference : str | None, default=None
        Reference level. Required for ``"all_vs_ref"`` mode.
    min_samples : int, default=2
        Minimum observations per group/subset. Comparisons with fewer
        observations are skipped.
    multitest_method : str, default="fdr_bh"
        Method for multiple testing correction across all comparisons
        (see :func:`statsmodels.stats.multipletests`).
    verbose : bool, default=True
        Whether to print progress messages.
    **kwargs
        Passed through to ``func``. Do not pass ``condition_key``,
        ``reference``, or ``contrast`` — these are set by ``compare()``.

    Returns
    -------
    pd.DataFrame
        Combined results with additional comparison columns. The ``padj``
        column is re-computed across all comparisons.

    Examples
    --------
    1-vs-all marker detection with logistic regression:

    >>> markers = dx.tl.compare(dx.tl.de, adata, condition_key="cell_type",
    ...                         mode="1_vs_all", method="lr")

    All pairwise comparisons:

    >>> pairwise = dx.tl.compare(dx.tl.de, adata, condition_key="cell_type",
    ...                          mode="pairwise", method="anova")

    All groups vs control:

    >>> vs_ctrl = dx.tl.compare(dx.tl.de, adata, condition_key="treatment",
    ...                         mode="all_vs_ref", reference="control")
    """
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    levels = sorted(adata.obs[condition_key].unique())
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 levels in '{condition_key}', got {len(levels)}")

    if mode == "all_vs_ref" and reference is None:
        raise ValueError("'reference' is required for 'all_vs_ref' mode")

    if reference is not None and reference not in levels:
        raise ValueError(f"Reference '{reference}' not found in '{condition_key}'. Available: {levels}")

    comparisons = _build_comparisons(levels, mode, reference)

    results = []
    for test_level, ref_level in comparisons:
        if mode == "1_vs_all":
            res = _run_one_vs_all(func, adata, condition_key, test_level, min_samples, verbose, **kwargs)
        else:
            res = _run_pairwise(func, adata, condition_key, test_level, ref_level, min_samples, verbose, **kwargs)

        if res is not None:
            results.append(res)

    if not results:
        raise ValueError("All comparisons failed. Check input data or set verbose=True.")

    results = pd.concat(results, axis=0).reset_index(drop=True)

    if "pval" in results.columns and results["pval"].notna().any():
        valid = results["pval"].notna()
        padj = sm.stats.multipletests(results.loc[valid, "pval"].values, method=multitest_method)[1]
        results["padj"] = np.nan
        results.loc[valid, "padj"] = padj

    sort_cols = [c for c in ["comparison", "test", "reference", "padj"] if c in results.columns]
    results = results.sort_values(by=sort_cols).reset_index(drop=True)

    return results


def _build_comparisons(
    levels: list[str],
    mode: str,
    reference: str | None,
) -> list[tuple[str, str | None]]:
    if mode == "1_vs_all":
        return [(level, None) for level in levels]
    elif mode == "pairwise":
        return [(l1, l2) for i, l1 in enumerate(levels) for l2 in levels[i + 1 :]]
    elif mode == "all_vs_ref":
        return [(level, reference) for level in levels if level != reference]
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: '1_vs_all', 'pairwise', 'all_vs_ref'")


def _run_one_vs_all(
    func: Callable,
    adata: AnnData,
    condition_key: str,
    test_level: str,
    min_samples: int,
    verbose: bool,
    **kwargs,
) -> pd.DataFrame | None:
    is_test = adata.obs[condition_key].values == test_level
    n_test = np.sum(is_test)
    n_rest = np.sum(~is_test)

    if n_test < min_samples or n_rest < min_samples:
        logger.warning(
            f"Skipping 1-vs-all for '{test_level}' ({n_test} vs {n_rest} samples)",
            verbose=verbose,
        )
        return None

    logger.info(f"Running 1-vs-all: {test_level} ({n_test}) vs rest ({n_rest})", verbose=verbose)

    a = adata.copy()
    a.obs["__compare__"] = is_test.astype(str)

    try:
        res = func(a, condition_key="__compare__", reference="False", contrast="True", **kwargs)
        res["comparison"] = test_level
        return res
    except Exception as e:
        logger.warning(f"DE failed for '{test_level}' vs rest: {e}", verbose=verbose)
        return None


def _run_pairwise(
    func: Callable,
    adata: AnnData,
    condition_key: str,
    test_level: str,
    ref_level: str,
    min_samples: int,
    verbose: bool,
    **kwargs,
) -> pd.DataFrame | None:
    mask = np.isin(adata.obs[condition_key].values, [test_level, ref_level])
    n_test = np.sum(adata.obs[condition_key].values[mask] == test_level)
    n_ref = np.sum(adata.obs[condition_key].values[mask] == ref_level)

    if n_test < min_samples or n_ref < min_samples:
        logger.warning(
            f"Skipping {test_level} vs {ref_level} ({n_test} vs {n_ref} samples)",
            verbose=verbose,
        )
        return None

    logger.info(f"Running: {test_level} vs {ref_level}", verbose=verbose)

    try:
        res = func(adata[mask, :].copy(), condition_key=condition_key, reference=ref_level, **kwargs)
        res["test"] = test_level
        res["reference"] = ref_level
        return res
    except Exception as e:
        logger.warning(f"DE failed for {test_level} vs {ref_level}: {e}", verbose=verbose)
        return None
