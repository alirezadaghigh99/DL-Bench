import numpy as np
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import lars_path as sklearn_lars_path

def lars_path(X, y, Xy=None, Gram=None, max_iter=500, alpha_min=0, method='lar',
              copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0,
              return_path=True, return_n_iter=False, positive=False):
    """
    Compute Least Angle Regression or Lasso path using the LARS algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Target values.

    Xy : array-like, shape (n_features,), optional
        Precomputed X.T @ y.

    Gram : array-like, shape (n_features, n_features), optional
        Precomputed Gram matrix (X.T @ X).

    max_iter : int, optional, default=500
        Maximum number of iterations to perform.

    alpha_min : float, optional, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter in the Lasso.

    method : {'lar', 'lasso'}, optional, default='lar'
        Specifies the type of path to compute. 'lar' is for Least Angle
        Regression, 'lasso' is for the Lasso.

    copy_X : bool, optional, default=True
        If True, X is copied; else, it may be overwritten.

    eps : float, optional, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors.

    copy_Gram : bool, optional, default=True
        If True, Gram is copied; else, it may be overwritten.

    verbose : int, optional, default=0
        Controls the verbosity; the higher, the more messages.

    return_path : bool, optional, default=True
        If True, returns the entire path; else, returns only the last point.

    return_n_iter : bool, optional, default=False
        If True, returns the number of iterations.

    positive : bool, optional, default=False
        Restrict coefficients to be >= 0.

    Returns
    -------
    alphas : array, shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.

    active : list
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas + 1)
        Coefficients along the path.

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is True.

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.linear_model import lars_path
    >>> X, y = datasets.make_regression(n_samples=100, n_features=10, noise=0.1)
    >>> alphas, active, coefs = lars_path(X, y, method='lasso')

    References
    ----------
    Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). "Least Angle
    Regression". Annals of Statistics.
    """
    X = check_array(X, copy=copy_X)
    y = check_array(y, ensure_2d=False)

    if Xy is None:
        Xy = safe_sparse_dot(X.T, y)

    alphas, active, coefs = sklearn_lars_path(
        X, y, Xy=Xy, Gram=Gram, max_iter=max_iter, alpha_min=alpha_min,
        method=method, copy_X=copy_X, eps=eps, copy_Gram=copy_Gram,
        verbose=verbose, return_path=return_path, positive=positive)

    if return_n_iter:
        n_iter = len(alphas) - 1
        return alphas, active, coefs, n_iter
    else:
        return alphas, active, coefs