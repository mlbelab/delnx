from ._glm_gp import (
    compute_gp_deviance,
    estimate_dispersion_mle,
    estimate_dispersion_moments,
    fit_beta_fisher_scoring,
    fit_beta_one_group,
)
from ._models import DispersionEstimator, LinearRegression, LogisticRegression, NegativeBinomialRegression
from ._quasi_likelihood import (
    compute_ql_dispersions,
    fit_dispersion_trend,
    ql_f_test,
    shrink_ql_dispersions,
)
