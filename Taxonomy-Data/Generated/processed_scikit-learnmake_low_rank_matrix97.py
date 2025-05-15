import numpy as np

def make_low_rank_matrix(n_samples, n_features, effective_rank, tail_strength=0.1, random_state=None):
    """
    Generate a mostly low-rank matrix with bell-shaped singular values.

    Parameters:
    - n_samples: int, number of samples (rows).
    - n_features: int, number of features (columns).
    - effective_rank: int, approximate rank of the matrix.
    - tail_strength: float, between 0 and 1, the relative importance of the tail of the singular values.
    - random_state: int or None, random seed for reproducibility.

    Returns:
    - X: ndarray of shape (n_samples, n_features), the generated low-rank matrix.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random Gaussian matrix
    U, _ = np.linalg.qr(np.random.randn(n_samples, n_samples))
    V, _ = np.linalg.qr(np.random.randn(n_features, n_features))

    # Generate singular values with a bell-shaped profile
    singular_values = np.exp(-np.linspace(0, 1, min(n_samples, n_features))**2 / (2 * (effective_rank / min(n_samples, n_features))**2))
    
    # Add tail strength
    tail = tail_strength * np.random.rand(min(n_samples, n_features))
    singular_values = (1 - tail_strength) * singular_values + tail

    # Construct the low-rank matrix
    S = np.zeros((n_samples, n_features))
    np.fill_diagonal(S, singular_values)
    X = U @ S @ V.T

    return X

