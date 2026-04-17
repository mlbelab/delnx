"""Effect size calculation functions for differential expression analysis.

This module provides functions to compute and evaluate effect sizes between condition
groups in RNA-seq data. It implements common effect size metrics including:

- Log2 fold change (log2FC): Quantifies expression differences between conditions
- Area under the ROC curve (AUROC): Measures classification performance between groups
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
from anndata import AnnData

from delnx._logging import logger
from delnx._typing import ComparisonMode, DataType
from delnx._utils import _get_layer, _to_dense

from ._utils import _infer_data_type, _validate_conditions


def _log2fc(
    X: np.ndarray,
    condition_mask: np.ndarray,
    data_type: DataType,
    eps: float = 1e-8,
) -> np.ndarray:
    """Calculate log2 fold changes between two conditions.

    This internal function computes log2 fold changes between test and reference conditions,
    handling different data types appropriately. For log-normalized data, it transforms
    back to linear space before calculating ratios to avoid issues with log-space arithmetic.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix of shape (n_samples, n_features).
    condition_mask : np.ndarray
        Boolean mask of shape (n_samples,) where:
        - True values indicate test condition samples
        - False values indicate reference condition samples
    data_type : DataType
        Type of expression data:
        - "counts": Raw count data, fold changes calculated directly
        - "lognorm": Log-normalized data (log1p transformed), automatically transformed
          back to linear space before calculating fold changes
        - "binary": Binary data (0/1 values), fold changes represent probability ratios
    eps : float, default=1e-8
        Small constant added to means to avoid division by zero.

    Returns
    -------
    np.ndarray
        Log2 fold changes for each feature, shape (n_features,). Positive values indicate
        higher expression in test condition, negative values indicate higher expression
        in reference condition.
    """
    if data_type not in ["counts", "lognorm", "binary"]:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Extract test and reference data once
    ref_data = X[~condition_mask, :]
    test_data = X[condition_mask, :]

    if data_type == "lognorm":
        # For log-normalized data (log1p transformed):
        ref_data = np.expm1(ref_data.astype(np.float64))
        test_data = np.expm1(test_data.astype(np.float64))

    ref_means = ref_data.mean(axis=0) + eps
    test_means = test_data.mean(axis=0) + eps
    log2fc = np.log2(test_means / ref_means)
    return np.asarray(log2fc, dtype=np.float64).flatten()


def log2fc(
    adata: AnnData,
    condition_key: str,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    min_samples: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate log2 fold changes between condition levels.

    This function computes log2 fold changes (log2FC) between different experimental
    conditions for all features in the dataset. It supports various comparison modes,
    different data types, and can normalize by size factors when appropriate.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and metadata.
    condition_key : str
        Column name in `adata.obs` containing condition labels.
    reference : str | tuple[str, str] | None, default=None
        Reference condition for comparisons, specified as:
        - Single string: reference condition for all comparisons
        - Tuple (reference, comparison): specific pair to compare
        - None: automatically determined based on mode parameter
    mode : ComparisonMode, default="all_vs_all"
        Comparison strategy:
        - "all_vs_ref": Compare all condition levels against reference level
        - "all_vs_all": Compare all pairs of condition levels
        - "1_vs_1": Compare only reference vs comparison (requires tuple reference)
    layer : str | None, default=None
        Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
    data_type : DataType, default="auto"
        Type of expression data:
        - "auto": Automatically infer from data characteristics
        - "counts": Raw count data
        - "lognorm": Log-normalized data (log1p of normalized counts)
        - "binary": Binary expression data (0/1)
    min_samples : int, default=2
        Minimum number of samples required per condition level.
        Comparisons with fewer samples are skipped.
    verbose : bool, default=True
        Whether to print progress information and data type inference results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing log2 fold change results with columns:
        - "feature": Feature/gene names
        - "test_condition": Test condition label
        - "ref_condition": Reference condition label
        - "log2fc": Log2 fold change values (positive means up-regulated in test condition)

    Examples
    --------
    Basic usage with automatic data type inference:

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("dataset.h5ad")
    >>> results = dx.tl.log2fc(adata, condition_key="treatment")

    Comparing specific conditions:

    >>> results = dx.tl.log2fc(adata, condition_key="treatment", reference=("control", "treated"), mode="1_vs_1")
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    comparisons = _validate_conditions(condition_values, reference, mode)

    # Get expression matrix and size factors
    X = _get_layer(adata, layer)

    # Infer data type if auto
    if data_type == "auto":
        data_type = _infer_data_type(X)
        logger.info(f"Inferred data type: {data_type}", verbose=verbose)
    else:
        logger.info(f"Using specified data type: {data_type}", verbose=verbose)

    # Calculate log2fc for each comparison
    results = []
    for group1, group2 in comparisons:
        # Get cell masks
        mask1 = adata.obs[condition_key].values == group1
        mask2 = adata.obs[condition_key].values == group2

        if np.sum(mask1) < min_samples or np.sum(mask2) < min_samples:
            logger.info(f"Skipping comparison {group1} vs {group2} with < {min_samples} samples", verbose=verbose)
            continue

        all_mask = mask1 | mask2

        # Get data for calculations
        group_data = X[all_mask, :]
        condition_mask = adata.obs.loc[all_mask, condition_key].values == group1

        # Calculate log2 fold change
        log2fc_values = _log2fc(
            X=group_data,
            condition_mask=condition_mask,
            data_type=data_type,
        )

        # Create results dataframe
        result_df = pd.DataFrame(
            {
                "feature": adata.var_names,
                "test_condition": group1,
                "ref_condition": group2,
                "log2fc": log2fc_values,
            }
        )

        results.append(result_df)

    if len(results) == 0:
        raise ValueError("No valid comparisons found for fold change analysis")

    # Combine results
    return pd.concat(results, axis=0)


@jax.jit
def _auroc(x: jnp.ndarray, groups: jnp.ndarray) -> float:
    """Calculate AUROC via Mann-Whitney U statistic.

    Computes AUROC as U / (n_pos * n_neg) where U is derived from the
    rank sum of positive samples. This is O(n log n) from the sort for
    ranking but avoids the more expensive ROC-curve construction.

    Parameters
    ----------
    x : jnp.ndarray
        Feature expression values, shape (n_samples,).
    groups : jnp.ndarray
        Binary group labels (1=test, 0=reference), shape (n_samples,).

    Returns
    -------
    float
        AUROC value in [0, 1].
    """
    n = x.shape[0]
    n_pos = jnp.sum(groups)
    n_neg = n - n_pos

    # Average rank (1-based) using argsort-of-argsort
    order = jnp.argsort(x)
    ranks = jnp.empty_like(x)
    ranks = ranks.at[order].set(jnp.arange(1, n + 1, dtype=x.dtype))

    # Handle ties: average ranks for identical values
    sorted_x = x[order]
    # Mark boundaries where values change
    same = jnp.concatenate([jnp.array([False]), sorted_x[1:] == sorted_x[:-1]])
    # Compute average ranks for tie groups via cumulative approach
    # For each run of identical values, replace with their mean rank
    avg_ranks = jnp.where(same, 0.0, ranks[order])
    # Forward-fill approach: use segment-based averaging
    # Simple approach: use the fact that for ties, rank = (first + last) / 2
    # which equals the average of the 1-based positions
    # scipy rankdata 'average' method — replicate via sorted values
    # Use a simpler approach: compute via sorted unique boundaries
    tie_adj_ranks = jnp.zeros_like(ranks)
    sorted_ranks = jnp.arange(1, n + 1, dtype=x.dtype)

    # Group consecutive equal values and assign mean rank
    # Identify group starts
    diff = jnp.concatenate([jnp.array([True]), sorted_x[1:] != sorted_x[:-1]])
    # Cumulative sum to get group IDs (0-based)
    group_ids = jnp.cumsum(diff) - 1

    # Sum of ranks per group and count per group
    n_groups = group_ids[-1] + 1
    group_sums = jnp.zeros(n, dtype=x.dtype).at[group_ids].add(sorted_ranks)
    group_counts = jnp.zeros(n, dtype=x.dtype).at[group_ids].add(1.0)

    # Mean rank per group, then broadcast back
    group_means = group_sums / jnp.maximum(group_counts, 1.0)
    avg_rank_sorted = group_means[group_ids]

    # Map back to original order
    final_ranks = jnp.empty_like(x)
    final_ranks = final_ranks.at[order].set(avg_rank_sorted)

    # Mann-Whitney U
    rank_sum_pos = jnp.sum(final_ranks * groups)
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    auroc = U / jnp.maximum(n_pos * n_neg, 1e-10)
    return jnp.clip(auroc, 0.0, 1.0)


_auroc_batch = jax.vmap(_auroc, in_axes=[1, None])


def _batched_auroc(
    X: np.ndarray,
    groups: np.ndarray,
    batch_size: int = 2048,
    verbose: bool = False,
) -> np.ndarray:
    """Run AUROC analysis in batches for efficient memory usage and performance.

    This internal function processes features in batches to calculate AUROC values
    efficiently, even for large datasets. It uses JAX's vectorized implementation
    for improved performance.

    Parameters
    ----------
    X : np.ndarray
        Expression data matrix of shape (n_samples, n_features). Can be dense or sparse.
    groups : np.ndarray
        Group labels of shape (n_samples,), where:
        - 1 indicates samples in the test condition
        - 0 indicates samples in the reference condition
    batch_size : int, default=2048
        Number of features to process per batch. Adjust based on available memory.
    verbose : bool, default=False
        Whether to show progress bar during batch processing.

    Returns
    -------
    np.ndarray
        Array of AUROC values for each feature, shape (n_features,).

    Notes
    -----
    This function leverages JAX's vectorized operations by using the vmapped
    version of the _auroc function, which significantly improves performance
    compared to sequential processing.
    """
    # Process in batches
    n_features = X.shape[1]

    # Convert groups to JAX array
    groups_jax = jnp.array(groups, dtype=jnp.int32)

    # Process all batches
    results = []
    for i in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        batch = slice(i, min(i + batch_size, n_features))
        X_batch = jnp.asarray(_to_dense(X[:, batch]), dtype=jnp.float32)

        # Calculate AUROC values for batch using vectorized function
        auroc_values = _auroc_batch(X_batch, groups_jax)

        results.append(auroc_values)

    # Concatenate results
    results = np.concatenate(results, axis=0)
    return results


def auroc(
    adata: AnnData,
    condition_key: str,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    min_samples: int = 2,
    batch_size: int = 2048,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate Area Under the Receiver Operating Characteristic (AUROC) between condition levels.

    This function computes AUROC values for all features between different experimental
    conditions. AUROC quantifies how well a feature's expression can distinguish between
    two conditions, providing a measure of the feature's discriminative power independent
    of any specific threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and metadata.
    condition_key : str
        Column name in `adata.obs` containing condition labels.
    reference : str | tuple[str, str] | None, default=None
        Reference condition for comparisons, specified as:
        - Single string: reference condition for all comparisons
        - Tuple (reference, comparison): specific pair to compare
        - None: automatically determined based on mode parameter
    mode : ComparisonMode, default="all_vs_all"
        Comparison strategy:
        - "all_vs_ref": Compare all condition levels against reference level
        - "all_vs_all": Compare all pairs of condition levels
        - "1_vs_1": Compare only reference vs comparison (requires tuple reference)
    layer : str | None, default=None
        Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
    min_samples : int, default=2
        Minimum number of samples required per condition level.
        Comparisons with fewer samples are skipped.
    batch_size : int, default=2048
        Number of features to process per batch. Adjust based on available memory
        and dataset size.
    verbose : bool, default=False
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing AUROC results with columns:
        - "feature": Feature/gene names
        - "test_condition": Test condition label
        - "ref_condition": Reference condition label
        - "auroc": AUROC values (0.5=random, 1=perfect separation with higher values in test)

    Examples
    --------
    Basic usage for all pairwise comparisons:

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("dataset.h5ad")
    >>> results = dx.tl.auroc(adata, condition_key="cell_type")

    Looking at specific condition comparisons:

    >>> # Compare only CD4+ T cells vs CD8+ T cells
    >>> results = dx.tl.auroc(adata, condition_key="cell_type", reference=("CD4+ T", "CD8+ T"), mode="1_vs_1")

    >>> # Compare all cell types against a reference type
    >>> results = dx.tl.auroc(adata, condition_key="cell_type", reference="B cells", mode="all_vs_ref")

    Notes
    -----
    - AUROC values range from 0 to 1, where:
      - 0.5 indicates the feature cannot distinguish between conditions (random)
      - Values >0.5 indicate higher expression in the test condition
      - Values <0.5 indicate higher expression in the reference condition
    - The implementation uses JAX for accelerated computation and batch processing
      to efficiently handle large datasets
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    comparisons = _validate_conditions(condition_values, reference, mode)

    # Get expression matrix
    X = _get_layer(adata, layer)

    # Calculate AUROC for each comparison
    results = []
    for group1, group2 in comparisons:
        # Get cell masks
        mask1 = adata.obs[condition_key].values == group1
        mask2 = adata.obs[condition_key].values == group2

        if np.sum(mask1) < min_samples or np.sum(mask2) < min_samples:
            logger.info(f"Skipping comparison {group1} vs {group2} with < {min_samples} samples", verbose=verbose)
            continue

        all_mask = mask1 | mask2

        # Get data for calculations
        data = X[all_mask, :]

        # Create binary groups vector (1 for test group, 0 for reference group)
        groups = (adata.obs.loc[all_mask, condition_key].values == group1).astype(np.int32)

        # Run batched AUROC calculation
        auroc_values = _batched_auroc(
            X=data,
            groups=groups,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Create results dataframe
        result_df = pd.DataFrame(
            {
                "feature": adata.var_names,
                "test_condition": group1,
                "ref_condition": group2,
                "auroc": auroc_values,
            }
        )

        results.append(result_df)

    if len(results) == 0:
        raise ValueError("No valid comparisons found for AUROC analysis")

    # Combine results
    return pd.concat(results, axis=0)
