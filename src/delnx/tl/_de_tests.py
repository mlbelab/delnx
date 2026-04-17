"""Statistical test functions for differential expression analysis."""

from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
from joblib import Parallel, delayed
from scipy import sparse, stats
from statsmodels.stats.anova import anova_lm

from delnx._logging import logger
from delnx._utils import _to_dense, suppress_output


def _run_lr_test(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariate_keys: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run logistic regression with likelihood ratio test for differential expression.

    This function fits two logistic regression models - a full model with the feature (gene)
    as a predictor, and a reduced model without the feature. It then performs a likelihood
    ratio test to assess if the feature significantly improves model fit.

    Parameters
    ----------
    X : np.ndarray
        Expression values for a single feature (gene), shape (n_samples,).
        Usually binary (0/1) or continuous values that work well with logistic regression.
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates, with one row per sample.
    condition_key : str
        Column name in model_data containing the condition labels.
    covariate_keys : list[str] | None, default=None
        Column names in model_data to include as covariates in both models.
    verbose : bool, default=False
        Whether to show statsmodels output during fitting.

    Returns
    -------
    tuple[float, float]
        Feature coefficient in the full model and p-value from likelihood ratio test.
    """
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariate_keys:
        covar_str = " + " + " + ".join(covariate_keys)

    # Fit models
    full_formula = f"{condition_key} ~ X{covar_str}"
    reduced_formula = f"{condition_key} ~ 1{covar_str}"

    with suppress_output(verbose):
        full_model = sm.Logit.from_formula(full_formula, data=model_data).fit(disp=False)
        reduced_model = sm.Logit.from_formula(reduced_formula, data=model_data).fit(disp=False)

    # LR test
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    pval = stats.chi2.sf(lr_stat, df_diff)

    return full_model.params["X"], pval



def _run_anova(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariate_keys: list[str] | None = None,
    method: str = "anova",
    verbose: bool = False,
) -> tuple[float, float]:
    """Run ANOVA or residual F-test for differential expression of continuous data.

    This function fits linear models and performs either standard ANOVA testing
    or residual F-tests. It's suitable for normalized expression data like log-transformed
    counts where linear models are appropriate.

    Parameters
    ----------
    X : np.ndarray
        Expression values for a single feature (gene), shape (n_samples,).
        Should contain continuous values, typically log-normalized expression.
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates, with one row per sample.
    condition_key : str
        Column name in model_data containing the condition labels.
    covariate_keys : list[str] | None, default=None
        Column names in model_data to include as covariates in both models.
    method : str, default="anova"
        Statistical approach to use:
        - "anova": Standard ANOVA testing on the condition term
        - "anova_residual": Residual F-test comparing residuals between models
    verbose : bool, default=False
        Whether to show statsmodels output during fitting.

    Returns
    -------
    tuple[float, float]
        Condition coefficient in the full model and p-value from the selected test.
    """
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariate_keys:
        covar_str = " + " + " + ".join(covariate_keys)

    null_formula = f"X ~ 1{covar_str}"
    formula = f"X ~ {condition_key}{covar_str}"

    with suppress_output(verbose):
        null_model = sm.OLS.from_formula(null_formula, data=model_data).fit()
        model = sm.OLS.from_formula(formula, data=model_data).fit()
        a0 = anova_lm(null_model)
        a1 = anova_lm(model)

    p_anova = a1.loc[condition_key, "PR(>F)"]
    p_resid_cdf = stats.f.cdf(
        a0.loc["Residual", "mean_sq"] / a1.loc["Residual", "mean_sq"],
        a0.loc["Residual", "df"],
        a1.loc["Residual", "df"],
    )
    p_resid = 1 - np.abs(0.5 - p_resid_cdf) * 2

    return model.params[condition_key], p_anova if method == "anova" else p_resid


def _run_binomial(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariate_keys: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run binomial GLM for binary expression data.

    This function fits a binomial generalized linear model with logit link,
    treating the expression values as the response variable and condition as
    a predictor. It's particularly suitable for binary expression data (0/1).

    Parameters
    ----------
    X : np.ndarray
        Binary expression values for a single feature (gene), shape (n_samples,).
        Should contain values that can be interpreted as binary outcomes (0/1).
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates, with one row per sample.
    condition_key : str
        Column name in model_data containing the condition labels.
    covariate_keys : list[str] | None, default=None
        Column names in model_data to include as covariates in the model.
    verbose : bool, default=False
        Whether to show statsmodels output during fitting.

    Returns
    -------
    tuple[float, float]
        Condition coefficient in the model and its associated p-value.

    Notes
    -----
    Unlike logistic regression which treats the condition as the outcome,
    this model treats the gene expression as the outcome, which is more
    appropriate when expression is truly binary.
    """
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariate_keys:
        covar_str = " + " + " + ".join(covariate_keys)

    # Use X as response and condition as predictor
    formula = f"X ~ {condition_key}{covar_str}"

    with suppress_output(verbose):
        # Fit model with binomial family and logit link
        glm = sm.GLM.from_formula(formula=formula, data=model_data, family=sm.families.Binomial()).fit(disp=False)

    return glm.params[condition_key], glm.pvalues[condition_key]


# Dictionary mapping backends to available test methods
METHODS = {
    "statsmodels": {
        "lr": _run_lr_test,
        "anova": _run_anova,
        "anova_residual": partial(_run_anova, method="anova_residual"),
        "binomial": _run_binomial,
    },
}


def _run_de(
    X: np.ndarray | sparse.spmatrix,
    model_data: pd.DataFrame,
    feature_names: pd.Index,
    condition_key: str,
    method: str,
    backend: str = "statsmodels",
    size_factors: np.ndarray | None = None,
    covariate_keys: list[str] | None = None,
    design_matrix: np.ndarray | None = None,
    test_idx: int | None = None,
    n_cpus: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run parallel GLM-based differential expression analysis for all features.

    This is the main executor function that coordinates differential expression testing
    across all features (genes) in parallel. It dynamically selects the appropriate
    statistical test based on the requested method and backend, and handles error recovery
    during the testing process.

    Parameters
    ----------
    X : np.ndarray | sparse.spmatrix
        Expression matrix of shape (n_samples, n_features). Can be dense or sparse.
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates, with one row per sample.
    feature_names : pd.Index
        Names of features/genes corresponding to columns in X.
    condition_key : str
        Column name in model_data containing the condition labels.
    method : str
        DE method to use:
        - "lr": Logistic regression with likelihood ratio test
        - "negbinom": Negative binomial regression for count data
        - "anova": ANOVA-based linear model
        - "anova_residual": Linear model with residual F-test
        - "binomial": Binomial GLM for binary data
    backend : str, default="statsmodels"
        Backend implementation to use for statistical tests:
        - "statsmodels": Python statsmodels implementation
        - "cuml": GPU-accelerated implementation (only for "lr" method)
    size_factors : np.ndarray | None, default=None
        Size factors for normalization, shape (n_samples,). Used only for "negbinom" method.
    covariate_keys : list[str] | None, default=None
        Column names in model_data to include as covariates in all models.
    n_cpus : int, default=1
        Number of parallel processes for feature testing. Use -1 to use all processors.
    verbose : bool, default=False
        Whether to show progress bar and warning messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with differential expression results containing:
        - "feature": Feature/gene names
        - "coef": Model coefficients
        - "pval": Raw p-values from statistical tests

    Notes
    -----
    - Failed tests are tracked and reported if verbose=True, but do not stop the process
    - For large datasets, adjust n_cpus and consider using a GPU-compatible backend
    - This function is called by the higher-level `de` function and shouldn't typically
      be used directly
    """
    # When a pre-built design matrix is provided, construct synthetic
    # model_data so existing statsmodels test functions work unchanged.
    if design_matrix is not None:
        if test_idx is None:
            raise ValueError("test_idx is required when design_matrix is provided")
        condition_key = "__test__"
        model_data = pd.DataFrame({condition_key: design_matrix[:, test_idx]})
        reduced_cols = [i for i in range(design_matrix.shape[1]) if i != test_idx]
        covariate_keys = [f"__cov_{i}__" for i in range(len(reduced_cols))]
        for cov_name, col_idx in zip(covariate_keys, reduced_cols):
            model_data[cov_name] = design_matrix[:, col_idx]

    # Choose test function for non-batched methods
    available_methods = METHODS.get(backend)
    if available_methods is None:
        raise ValueError(f"Backend '{backend}' not supported.")

    test_func = available_methods.get(method)
    if test_func is None:
        raise ValueError(
            f"Method '{method}' not supported for backend '{backend}'. Available methods: {list(available_methods.keys())}"
        )

    if method == "negbinom":
        test_func = partial(test_func, size_factors=size_factors)

    def _process_feature(i: int) -> tuple[str, float, float] | tuple[str, None, None]:
        """Process a single feature and return test results or None if test failed.

        This inner function handles testing for an individual feature, including
        error handling to ensure one failing test doesn't stop the entire process.

        Parameters
        ----------
        i : int
            Index of the feature in the expression matrix X.

        Returns
        -------
        tuple[str, float, float] | tuple[str, None, None]
            If successful: (feature_name, coefficient, p-value)
            If failed: (feature_name, None, None)
        """
        try:
            x = _to_dense(X[:, i]).flatten()
            coef, pval = test_func(
                x, model_data, condition_key=condition_key, covariate_keys=covariate_keys, verbose=False
            )
            return feature_names[i], coef, pval
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error testing feature {feature_names[i]}: {str(e)}", verbose=verbose)
            return feature_names[i], None, None

    # Run tests in parallel with progress bar
    n_features = X.shape[1]

    # Process all features in parallel with a progress bar using joblib's generator return
    feature_results = list(
        tqdm.tqdm(
            Parallel(n_jobs=n_cpus, return_as="generator")(delayed(_process_feature)(i) for i in range(n_features)),
            total=n_features,
            disable=not verbose,
        )
    )

    # Collect successful results and count errors
    results = {
        "feature": [],
        "coef": [],
        "pval": [],
    }
    errors = {}

    for feat, coef, pval in feature_results:
        if coef is not None and pval is not None:
            results["feature"].append(feat)
            results["coef"].append(coef)
            results["pval"].append(pval)
        else:
            errors[feat] = "Test failed"

    if len(errors) > 0 and verbose:
        logger.warning(
            f"DE analysis failed for {len(errors)} features: {list(errors.keys())}",
            verbose=verbose,
        )

    results_df = pd.DataFrame(results)
    return results_df
