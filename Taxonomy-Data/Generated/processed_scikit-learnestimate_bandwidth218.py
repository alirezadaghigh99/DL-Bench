import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """
    Estimate the bandwidth to use with the mean-shift algorithm.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
        Input points.
    - quantile: float, default=0.3
        The quantile of the nearest neighbor distances to use.
    - n_samples: int, default=None
        The number of samples to use. If None, all samples are used.
    - random_state: int, default=0
        The seed of the pseudo random number generator to use when shuffling the data.
    - n_jobs: int, default=None
        The number of parallel jobs to run for neighbors search. None means 1.

    Returns:
    - bandwidth: float
        The estimated bandwidth.
    """
    if n_samples is not None:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(X), size=n_samples, replace=False)
        X = X[indices]

    # Fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=int(len(X) * quantile), n_jobs=n_jobs)
    nbrs.fit(X)

    # Find the distances to the nearest neighbors
    distances, _ = nbrs.kneighbors(X)

    # Use the quantile of the distances as the bandwidth
    bandwidth = np.quantile(distances[:, -1], quantile)
    return bandwidth

