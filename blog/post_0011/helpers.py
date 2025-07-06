import numpy as np


# -------------------------------------------------------------------------
#  Sample vectors
# -------------------------------------------------------------------------
def sample_vectors(n: int, m: int, normalized: bool = False) -> list[np.ndarray]:
    """Return m vectors in n dimensions, with each vector element sampled from a standard-normal distribution."""
    return [sample_vector(n, normalized) for _ in range(m)]


def sample_vector(n: int, normalized: bool = False) -> np.ndarray:
    """Return a vector in n dimensions, with each vector element sampled from a standard-normal distribution."""

    # sample vector
    vector = np.random.normal(loc=0, scale=1, size=n)

    # normalize vector if requested
    if normalized:
        vector = vector / np.linalg.norm(vector)

    # return
    return vector


# -------------------------------------------------------------------------
#  Metrics
# -------------------------------------------------------------------------
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""
    return float(np.linalg.norm(v1 - v2))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the cosine similarity between two vectors."""
    norm_v1 = float(np.linalg.norm(v1))
    norm_v2 = float(np.linalg.norm(v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))
