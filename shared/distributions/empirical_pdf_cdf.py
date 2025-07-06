"""
Functions to compute empirical PDF and CDF from a given dataset.
"""

import math

import numpy as np
from scipy.stats import norm


# =================================================================================================
#  Empirical PDF / CDF
# =================================================================================================
def empirical_pdf(
    samples: list[float] | np.ndarray, x_values: list[float] | np.ndarray, max_n_kernels: int = 1000
) -> list[float]:
    """
    Evaluate PDF of the distribution represented by provided samples at given x_values,
    using Kernel Density Estimation (KDE) with automatically selected parameters and Gaussian kernels.

    Result is returned as a list of floats.
    """

    # --- prep ----------------------------------------
    n_kernels = min(len(samples), max_n_kernels)
    k_mu, k_std = _compute_kde_mu_std(samples, n_kernels)

    # --- compute -------------------------------------
    return [float(sum(norm.pdf([x - mu for mu in k_mu], loc=0, scale=k_std))) / n_kernels for x in x_values]


def empirical_cdf(samples: list[float] | np.ndarray, x_values: list[float] | np.ndarray, max_n_kernels: int = 1000):
    """
    Evaluate CDF of the distribution represented by provided samples at given x_values,
    using Kernel Density Estimation (KDE) with automatically selected parameters and Gaussian kernels.

    Result is returned as a list of floats.
    """

    # --- prep ----------------------------------------
    n_kernels = min(len(samples), max_n_kernels)
    k_mu, k_std = _compute_kde_mu_std(samples, n_kernels)

    # --- compute -------------------------------------
    return [float(sum(norm.cdf([x - mu for mu in k_mu], loc=0, scale=k_std))) / n_kernels for x in x_values]


# =================================================================================================
#  Internal helpers
# =================================================================================================
def _compute_kde_mu_std(samples: list[float] | np.ndarray, n_kernels: int) -> tuple[list[float], float]:
    # --- prep --------------------------------------------
    sorted_samples = sorted(samples)
    n_samples = len(sorted_samples)

    # --- compute -----------------------------------------
    k_mu = [sorted_samples[int((i + 0.5) * (n_samples / n_kernels))] for i in range(n_kernels)]
    k_std = max(3.0, math.sqrt(n_samples)) * float(np.median(np.diff(k_mu)))

    # --- return ------------------------------------------
    return k_mu, k_std
