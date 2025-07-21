import time
from functools import cache

import torch

from .optimization import VectorOptimizer, max_abs_scaled_dot_product


def find_max_n_vectors(
    n_dims: int,
    epsilon: float,
) -> tuple[torch.Tensor, float]:
    """
    Find maximum n_vectors that fit in n_dims with epsilon orthogonality tolerance.
    """

    # --- initialize bisection ----------------------------
    n_min = n_dims  # these fit for sure
    n_max = n_dims + 1
    while do_vectors_fit(n_dims=n_dims, n_vectors=n_max, epsilon=epsilon):
        n_max = n_dims + 2 * (n_max - n_dims)

    # --- bisection ---------------------------------------
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        if do_vectors_fit(n_dims=n_dims, n_vectors=n_mid, epsilon=epsilon):
            n_min = n_mid
        else:
            n_max = n_mid

    # --- done --------------------------------------------
    return n_min  # this is the largest n_vectors that fit, since n_max=n_min+1 does not fit


@cache
def do_vectors_fit(n_dims: int, n_vectors: int, epsilon: float) -> bool:
    """
    Check if we can fit n_vectors vectors in n_dims dimensions such that they are orthogonal with tolerance epsilon.
    """
    t_start = time.time_ns()
    print(f"do_vectors_fit(n_dims={n_dims:>5}, n_vectors={n_vectors:>5}, epsilon={epsilon:.3f}) --> ", end="")
    success = max_non_orthogonality(n_dims=n_dims, n_vectors=n_vectors) <= epsilon
    t_elapsed_sec = (time.time_ns() - t_start) / 1e9
    print(f"{success}".ljust(10) + f" ({t_elapsed_sec:6.1f} sec)")

    return success


@cache
def max_non_orthogonality(n_dims: int, n_vectors: int) -> float:
    """
    Compute the maximum absolute scaled dot product of the vectors in v.
    """
    if n_vectors <= n_dims:
        return 0.0
    else:
        use_float64 = (n_dims + n_vectors) <= 275  # only use GPU when it results in speedups
        optimizer = VectorOptimizer(n_dims=n_dims, n_vectors=n_vectors, alpha=20.0, use_float64=use_float64)
        optimizer.solve(verbose=False)
        return max_abs_scaled_dot_product(optimizer.v)
