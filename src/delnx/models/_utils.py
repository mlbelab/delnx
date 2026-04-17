"""JAX utility functions for negative binomial model fitting."""

import jax.numpy as jnp


def safe_slogdet(matrix):
    """Robust slogdet computation for JAX with regularization"""
    eye = jnp.eye(matrix.shape[-1])
    reg_matrix = matrix + 1e-12 * eye

    sign, logdet = jnp.linalg.slogdet(reg_matrix)

    logdet = jnp.where(jnp.isfinite(logdet), logdet, -1e10)
    sign = jnp.where(jnp.isfinite(sign), sign, 0.0)

    return sign, logdet
