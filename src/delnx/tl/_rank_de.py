"""Rank-based differential expression analysis for single-cell data.

This module provides rank-based differential expression analysis using Area Under the
ROC Curve (AUROC) statistics. It efficiently handles sparse gene expression matrices
by implementing optimized ranking algorithms with optional tie correction for improved
statistical accuracy.
"""

import jax
import jax.numpy as jnp
import numba
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
from anndata import AnnData
from scipy import sparse, stats

from delnx._logging import logger
from delnx._utils import _get_layer

# =============================================================================
# Core Numba-optimized ranking functions
# =============================================================================


@numba.njit
def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Fast 1D ranking with average tie handling."""
    n = len(arr)
    if n <= 1:
        return np.array([1.0]) if n == 1 else np.empty(0, dtype=np.float64)

    sorter = np.argsort(arr)
    arr = arr[sorter]
    obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))
    dense = np.empty(obs.size, dtype=np.int64)
    dense[sorter] = obs.cumsum()
    # cumulative counts of each unique value
    count = np.concatenate((np.flatnonzero(obs), np.array([len(obs)])))
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1)
    return ranks


@numba.njit
def _rankdata_with_ties(arr: np.ndarray) -> tuple[np.ndarray, np.float64]:
    """Fast 1D ranking with tie correction calculation."""
    n = len(arr)
    if n <= 1:
        return (np.array([1.0]) if n == 1 else np.empty(0, dtype=np.float64), 1.0)

    sorter = np.argsort(arr)
    sorted_arr = arr[sorter]

    # Find tie group boundaries more efficiently
    tie_starts = [0]
    for i in range(1, n):
        if sorted_arr[i] != sorted_arr[i - 1]:
            tie_starts.append(i)
    tie_starts.append(n)

    # Calculate ranks and tie correction simultaneously
    ranks = np.empty(n, dtype=np.float64)
    tie_sum = 0.0

    for i in range(len(tie_starts) - 1):
        start, end = tie_starts[i], tie_starts[i + 1]
        tie_size = end - start
        avg_rank = (start + end + 1) / 2.0

        # Assign average rank to tied group
        for j in range(start, end):
            ranks[sorter[j]] = avg_rank

        # Accumulate tie contribution
        if tie_size > 1:
            tie_sum += tie_size**3 - tie_size

    # Calculate tie correction
    tie_correction = 1.0 - tie_sum / (n**3 - n) if n >= 2 else 1.0
    return ranks, tie_correction


@numba.njit(parallel=True, cache=True)
def _rank_sparse_batch_parallel(
    data: np.ndarray, indptr: np.ndarray, nrows: int, ncols: int, use_ties: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized parallel ranking for sparse matrix batches."""
    ranked_data = np.empty_like(data, dtype=np.float64)
    zero_ranks = np.zeros(ncols, dtype=np.float64)
    tie_corrections = np.ones(ncols, dtype=np.float64)

    for col_idx in numba.prange(ncols):
        start_idx = indptr[col_idx]
        end_idx = indptr[col_idx + 1]
        n_nonzero = end_idx - start_idx

        if n_nonzero == 0:
            zero_ranks[col_idx] = 0.0
            continue

        n_zero = nrows - n_nonzero
        zero_ranks[col_idx] = (n_zero + 1) / 2.0 if n_zero > 0 else 0.0

        col_data = data[start_idx:end_idx]

        if use_ties:
            nonzero_ranks, nonzero_tie_corr = _rankdata_with_ties(col_data)
            # Calculate combined tie correction including zeros
            if nrows >= 2:
                zero_contrib = n_zero**3 - n_zero if n_zero > 1 else 0.0
                nonzero_contrib = (1.0 - nonzero_tie_corr) * (n_nonzero**3 - n_nonzero)
                total_contrib = zero_contrib + nonzero_contrib
                tie_corrections[col_idx] = 1.0 - total_contrib / (nrows**3 - nrows)
        else:
            nonzero_ranks = _rankdata(col_data)

        ranked_data[start_idx:end_idx] = nonzero_ranks + n_zero

    return ranked_data, zero_ranks, tie_corrections


def _rank_sparse_batch_serial(
    data: np.ndarray, indptr: np.ndarray, nrows: int, ncols: int, use_ties: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serial ranking for sparse matrix batches using scipy.stats.rankdata."""
    ranked_data = np.empty_like(data, dtype=np.float64)
    zero_ranks = np.zeros(ncols, dtype=np.float64)
    tie_corrections = np.ones(ncols, dtype=np.float64)

    for col_idx in range(ncols):
        start_idx = indptr[col_idx]
        end_idx = indptr[col_idx + 1]

        if end_idx > start_idx:  # if column has non-zero elements
            n_nonzero = end_idx - start_idx
            n_zero = nrows - n_nonzero

            # Calculate zero rank (average of ranks 1 to n_zero)
            zero_ranks[col_idx] = (n_zero + 1) / 2.0 if n_zero > 0 else 0.0

            # Rank non-zero values
            col_data = data[start_idx:end_idx]
            nonzero_ranks = stats.rankdata(col_data, method="average")
            ranked_data[start_idx:end_idx] = nonzero_ranks + n_zero

        else:
            # Empty column case
            zero_ranks[col_idx] = 0.0

    return ranked_data, zero_ranks, tie_corrections


# =============================================================================
# JAX AUROC calculation
# =============================================================================


@jax.jit
def _auroc_batch_with_ties(
    X: jnp.ndarray, one_hot: jnp.ndarray, zero_ranks: jnp.ndarray, tie_corrections: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optimized AUROC calculation with tie correction."""
    # Reconstruct full ranks efficiently
    nonzero_mask = X > 0
    full_ranks = jnp.where(nonzero_mask, X, zero_ranks[None, :])

    # Compute statistics
    n_pos = jnp.maximum(one_hot.sum(axis=0), 1e-10)
    n_neg = jnp.maximum(X.shape[0] - n_pos, 1e-10)

    # Mann-Whitney U statistic
    rank_sum_pos = full_ranks.T @ one_hot
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2

    # AUROC and p-values
    aucs = jnp.clip(U / (n_pos * n_neg), 0.0, 1.0)

    # Tie-corrected p-values
    mu_U = n_pos * n_neg / 2
    sigma_U_sq = (n_pos * n_neg * (n_pos + n_neg + 1) / 12) * tie_corrections[:, None]
    sigma_U = jnp.sqrt(jnp.maximum(sigma_U_sq, 1e-20))

    z_score = jnp.where(sigma_U > 1e-10, jnp.abs(U - mu_U) / sigma_U, 0.0)

    return aucs, z_score


# =============================================================================
# Batch processing function
# =============================================================================


def _process_batch(
    X_batch: sparse.csc_matrix, one_hot: jnp.ndarray, use_ties: bool, use_parallel: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Process a single batch with optimized ranking and AUROC calculation."""
    # Ensure data is writable
    X_batch.data.setflags(write=True)

    # Choose ranking algorithm
    rank_func = _rank_sparse_batch_parallel if use_parallel else _rank_sparse_batch_serial
    ranked_data, zero_ranks, tie_corrections = rank_func(
        X_batch.data, X_batch.indptr, X_batch.shape[0], X_batch.shape[1], use_ties
    )

    # Convert to JAX arrays efficiently
    X_batch.data = ranked_data
    X_dense = jnp.asarray(X_batch.toarray(), dtype=jnp.float32)
    zero_ranks_jax = jnp.asarray(zero_ranks, dtype=jnp.float32)
    tie_corrections_jax = jnp.asarray(tie_corrections, dtype=jnp.float32)

    aucs, z_scores = _auroc_batch_with_ties(X_dense, one_hot, zero_ranks_jax, tie_corrections_jax)
    pvals = 2 * jax.scipy.stats.norm.sf(z_scores)

    # Single device-to-host transfer
    return np.asarray(aucs), np.asarray(pvals), np.asarray(z_scores)


def _validate_inputs(adata: AnnData, condition_key: str, min_samples: int) -> list[str]:
    """Validate inputs and return valid conditions."""
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    condition_values = adata.obs[condition_key].values
    unique_conditions = np.unique(condition_values)

    valid_conditions = []
    for condition in unique_conditions:
        n_samples = np.sum(condition_values == condition)
        if n_samples >= min_samples:
            valid_conditions.append(condition)

    if len(valid_conditions) < 2:
        raise ValueError(f"Need at least 2 valid conditions, found {len(valid_conditions)}")

    return valid_conditions


def _determine_algorithm(n_cpus: int, use_ties: bool, verbose: bool) -> tuple[bool, int]:
    """Set threads and determine whether to use parallel processing."""
    if n_cpus is None:
        n_cpus = numba.get_num_threads()
    else:
        max_threads = numba.get_num_threads()
        if n_cpus > max_threads:
            logger.warning(f"Requested n_cpus={n_cpus} exceeds available {max_threads}, using {max_threads}")
            n_cpus = max_threads
        numba.set_num_threads(n_cpus)

    # Decide on parallel processing
    # Serial processing (without ties) tends to be faster for small CPU counts
    use_parallel = (n_cpus >= 4) or use_ties

    return use_parallel


# =============================================================================
# Main user-facing function
# =============================================================================


def rank_de(
    adata: AnnData,
    condition_key: str,
    layer: str | None = None,
    use_ties: bool = False,
    multitest_method: str = "fdr_bh",
    n_cpus: int | None = None,
    min_samples: int = 2,
    batch_size: int | None = 512,
    verbose: bool = True,
) -> pd.DataFrame:
    """Perform rank-based differential expression analysis using AUROC statistics.

    This function identifies differentially expressed features between conditions using
    Area Under the ROC Curve (AUROC) analysis on ranked gene expression data. It
    efficiently handles sparse matrices with memory-optimized batch processing and
    provides optional tie correction for improved statistical accuracy.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and metadata.
    condition_key : str
        Column name in `adata.obs` containing condition labels.
    layer : str | None, default=None
        Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
    use_ties : bool, default=False
        Whether to apply tie correction to p-value calculations. Recommended for
        sparse data with many identical values (especially zeros).
    multitest_method : str, default='fdr_bh'
        Method for multiple testing correction. Accepts any method supported by :func:`statsmodels.stats.multipletests`. Common options include:
            - "fdr_bh": Benjamini-Hochberg FDR correction
            - "bonferroni": Bonferroni correction
    n_cpus : int | None, default=None
        Number of CPU cores for parallel processing. If None, uses available threads.
        Parallel processing is enabled when n_cpus >= 4 or use_ties=True.
    min_samples : int, default=2
        Minimum samples required per condition. Conditions with fewer samples excluded.
    batch_size : int | None, default=512
        Features per batch. Larger values use more memory but may be more efficient. If None, processes all features at once.
    verbose : bool, default=True
        Whether to show progress and algorithm information.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - "feature": Feature/gene names
        - "condition": Condition label (one-vs-all comparison)
        - "auroc": AUROC values (0.5=random, >0.5=upregulated, <0.5=downregulated)
        - "pval": Two-tailed p-values from Mann-Whitney U test
        - "tie_corrected": Whether tie correction was applied

    Examples
    --------
    Basic one-vs-all differential expression:

    >>> import delnx as dx
    >>> results = dx.tl.rank_de(adata, condition_key="cell_type")

    With tie correction for improved p-values:

    >>> results = dx.tl.rank_de(adata, condition_key="treatment", use_ties=True)

    Custom batch size for memory optimization:

    >>> results = dx.tl.rank_de(adata, condition_key="condition", batch_size=1024)

    Notes
    -----
    This implementation uses several optimizations:
    - Numba JIT compilation for fast ranking algorithms
    - JAX acceleration for AUROC calculations
    - Memory-efficient batch processing to handle large datasets
    - Automatic algorithm selection based on data characteristics
    - Optional tie correction for statistical accuracy with sparse data
    """
    # Input validation
    valid_conditions = _validate_inputs(adata, condition_key, min_samples)
    use_parallel = _determine_algorithm(n_cpus, use_ties, verbose)

    # Get data and encode conditions
    X = _get_layer(adata, layer)
    if not sparse.issparse(X):
        X = sparse.csc_matrix(X)
    elif not isinstance(X, sparse.csc_matrix):
        X = X.tocsc()

    condition_values = adata.obs[condition_key].values
    groups = np.zeros(len(condition_values), dtype=np.int32)
    for i, condition in enumerate(valid_conditions):
        groups[condition_values == condition] = i

    # Pre-compute one-hot encoding for efficiency
    n_classes = len(valid_conditions)
    one_hot = jax.nn.one_hot(jnp.array(groups, dtype=jnp.int32), n_classes, dtype=jnp.float32)

    if verbose:
        logger.info(f"Processing {X.shape[1]} features across {len(valid_conditions)} conditions", verbose=True)

    # Batch processing
    batch_size = batch_size or X.shape[1]
    n_features = X.shape[1]
    all_aucs, all_pvals, all_z_scores = [], [], []

    for start_idx in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        end_idx = min(start_idx + batch_size, n_features)
        X_batch = X[:, start_idx:end_idx].copy()

        aucs, pvals, z_scores = _process_batch(X_batch, one_hot, use_ties, use_parallel)
        all_aucs.append(aucs)
        all_pvals.append(pvals)
        all_z_scores.append(z_scores)

    # Combine results and create output DataFrame
    final_aucs = np.concatenate(all_aucs, axis=0)
    final_pvals = np.concatenate(all_pvals, axis=0)
    final_z_scores = np.concatenate(all_z_scores, axis=0)

    # Reshape to long format
    n_features, n_conditions = final_aucs.shape
    features_long = np.tile(adata.var_names.values, n_conditions)
    conditions_long = np.repeat(valid_conditions, n_features)
    aucs_long = final_aucs.T.flatten()
    pvals_long = final_pvals.T.flatten()
    z_scores_long = final_z_scores.T.flatten()

    results = pd.DataFrame(
        {
            "feature": features_long,
            "condition": conditions_long,
            "auroc": aucs_long,
            "z_score": z_scores_long,
            "pval": pvals_long,
        }
    )

    # Perform multiple testing correction
    padj = sm.stats.multipletests(
        results["pval"][results["pval"].notna()].values,
        method=multitest_method,
    )[1]
    results["padj"] = np.nan  # Initialize with NaN
    results.loc[results["pval"].notna(), "padj"] = padj
    results["pval"] = np.clip(results["pval"], 1e-50, 1)
    results["padj"] = np.clip(results["padj"], 1e-50, 1)

    results = results.sort_values(by=["condition", "auroc", "pval"], ascending=[True, False, True]).reset_index(
        drop=True
    )

    return results
