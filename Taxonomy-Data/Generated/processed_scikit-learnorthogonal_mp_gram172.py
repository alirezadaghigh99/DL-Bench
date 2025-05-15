import numpy as np
from numpy.linalg import norm, lstsq

def orthogonal_mp(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    X = np.array(X, copy=copy_X)
    y = np.array(y, copy=False)
    
    n_samples, n_features = X.shape
    
    if y.ndim == 1:
        y = y[:, np.newaxis]
    
    n_targets = y.shape[1]
    
    if n_nonzero_coefs is None:
        n_nonzero_coefs = max(1, int(0.1 * n_features))
    
    if n_nonzero_coefs > n_features:
        raise ValueError("Number of non-zero coefficients cannot exceed the number of features.")
    
    if tol is not None:
        n_nonzero_coefs = n_features  # Allow full selection if tol is specified
    
    if precompute == 'auto':
        precompute = n_targets > 1 and n_samples > n_features
    
    if precompute:
        Gram = np.dot(X.T, X)
    else:
        Gram = None
    
    coef = np.zeros((n_features, n_targets))
    residuals = y.copy()
    index_set = []
    path = []
    
    for k in range(n_nonzero_coefs):
        if precompute:
            correlations = np.dot(X.T, residuals)
        else:
            correlations = np.dot(X.T, residuals)
        
        if n_targets == 1:
            correlations = correlations.ravel()
        
        best_index = np.argmax(np.abs(correlations), axis=0)
        
        if np.all(correlations[best_index] == 0):
            break
        
        index_set.append(best_index)
        
        if precompute:
            A = Gram[np.ix_(index_set, index_set)]
            b = np.dot(X[:, index_set].T, y)
        else:
            A = np.dot(X[:, index_set].T, X[:, index_set])
            b = np.dot(X[:, index_set].T, y)
        
        coef_subset, _, _, _ = lstsq(A, b, rcond=None)
        
        residuals = y - np.dot(X[:, index_set], coef_subset)
        
        if tol is not None and norm(residuals) ** 2 <= tol:
            break
        
        if return_path:
            full_coef = np.zeros((n_features, n_targets))
            full_coef[index_set] = coef_subset
            path.append(full_coef)
    
    coef[index_set] = coef_subset
    
    if return_path:
        path.append(coef.copy())
    
    if return_n_iter:
        return (coef if n_targets > 1 else coef.ravel(), len(index_set))
    
    return coef if n_targets > 1 else coef.ravel()

