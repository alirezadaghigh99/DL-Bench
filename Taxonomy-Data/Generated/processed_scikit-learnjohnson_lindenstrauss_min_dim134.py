import numpy as np

def johnson_lindenstrauss_min_dim(n_samples, eps):
    """
    Calculate the minimum number of components required for a random projection
    to ensure an eps-embedding with good probability, according to the
    Johnson-Lindenstrauss lemma.

    Parameters:
    n_samples (int or array-like): The number of samples.
    eps (float or array-like): The maximum distortion rate (0 < eps < 1).

    Returns:
    int or np.ndarray: The minimal number of components required.
    """
    n_samples = np.atleast_1d(n_samples)
    eps = np.atleast_1d(eps)

    if np.any(eps <= 0) or np.any(eps >= 1):
        raise ValueError("eps must be between 0 and 1 (exclusive).")

    min_dim = (4 * np.log(n_samples)) / (eps**2 / 2 - eps**3 / 3)
    return np.ceil(min_dim).astype(int)

