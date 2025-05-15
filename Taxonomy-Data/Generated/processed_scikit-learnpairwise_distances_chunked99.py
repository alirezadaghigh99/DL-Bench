from sklearn.metrics import pairwise_distances
import numpy as np
import math

def pairwise_distances_chunked(X, Y=None, reduce_func=None, metric='euclidean', n_jobs=1, working_memory=1024, **kwds):
    """
    Generate a distance matrix chunk by chunk with optional reduction.

    Parameters:
    - X: array-like of shape (n_samples_X, n_features)
    - Y: array-like of shape (n_samples_Y, n_features), optional
    - reduce_func: callable, optional
        Function to apply to each chunk of the distance matrix.
    - metric: str or callable, default='euclidean'
    - n_jobs: int, default=1
    - working_memory: int, default=1024
        The maximum amount of memory to use for a chunk (in MB).
    - **kwds: additional keyword arguments for the distance computation.

    Yields:
    - A contiguous slice of the distance matrix, optionally processed by reduce_func.
    """
    if Y is None:
        Y = X

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    # Estimate the size of a single distance matrix element in bytes
    element_size = np.dtype(np.float64).itemsize
    # Calculate the number of elements that fit in the working memory
    max_elements = (working_memory * (1024 ** 2)) // element_size
    # Calculate the number of rows to process in each chunk
    chunk_size = max(1, int(max_elements // n_samples_Y))

    for start in range(0, n_samples_X, chunk_size):
        end = min(start + chunk_size, n_samples_X)
        X_chunk = X[start:end]
        distances = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=n_jobs, **kwds)

        if reduce_func is not None:
            yield reduce_func(distances)
        else:
            yield distances

