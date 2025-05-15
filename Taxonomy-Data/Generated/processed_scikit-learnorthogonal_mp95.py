import numpy as np
from numpy.linalg import norm, lstsq

def orthogonal_mp(X, y, n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False):
    """
    Orthogonal Matching Pursuit (OMP) algorithm.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
    - y: ndarray of shape (n_samples,) or (n_samples, n_targets)
    - n_nonzero_coefs: int, optional, default=None
    - tol: float, optional, default=None
    - precompute: bool, optional, default=False
    - copy_X: bool, optional, default=True
    - return_path: bool, optional, default=False
    - return_n_iter: bool, optional, default=False

    Returns:
    - coef: ndarray of shape (n_features,) or (n_features, n_targets)
    - n_iters: int, optional, number of active features across every target
    """
    if copy_X:
        X = X.copy()
    if y.ndim == 1:
        y = y[:, np.newaxis]

    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    if n_nonzero_coefs is None and tol is None:
        raise ValueError("Either n_nonzero_coefs or tol must be provided.")

    if n_nonzero_coefs is not None and tol is not None:
        raise ValueError("Only one of n_nonzero_coefs or tol should be provided.")

    if n_nonzero_coefs is None:
        n_nonzero_coefs = n_features

    coef = np.zeros((n_features, n_targets))
    residuals = y.copy()
    indices = []
    path = []

    for k in range(n_nonzero_coefs):
        if tol is not None and norm(residuals) ** 2 <= tol:
            break

        # Compute correlations
        correlations = X.T @ residuals
        if n_targets == 1:
            correlations = correlations.flatten()

        # Select the feature with the maximum correlation
        new_idx = np.argmax(np.abs(correlations), axis=0)
        indices.append(new_idx)

        # Update the selected features
        X_selected = X[:, indices]

        # Solve least squares problem
        coef_selected, _, _, _ = lstsq(X_selected, y, rcond=None)

        # Update residuals
        residuals = y - X_selected @ coef_selected

        # Store the path
        if return_path:
            coef_path = np.zeros((n_features, n_targets))
            coef_path[indices] = coef_selected
            path.append(coef_path)

    # Final coefficients
    coef[indices] = coef_selected

    if return_path:
        path = np.array(path)

    if return_n_iter:
        return coef, len(indices)
    elif return_path:
        return coef, path
    else:
        return coef

