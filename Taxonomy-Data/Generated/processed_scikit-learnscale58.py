import numpy as np
from scipy import sparse

def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """
    Standardize a dataset along any axis.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.
    """
    if sparse.issparse(X):
        raise TypeError("Sparse matrices are not supported by this function.")

    X = np.array(X, copy=copy)

    if with_mean:
        mean = np.mean(X, axis=axis, keepdims=True)
        X -= mean

    if with_std:
        std = np.std(X, axis=axis, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        X /= std

    return X

