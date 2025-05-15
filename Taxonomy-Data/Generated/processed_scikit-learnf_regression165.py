import numpy as np
from scipy import stats

def r_regression(X, y, center=True):
    """Calculate the Pearson correlation coefficient between each feature in X and the target y."""
    X = np.asarray(X)
    y = np.asarray(y)
    
    if center:
        X = X - np.mean(X, axis=0)
        y = y - np.mean(y)
    
    corr = np.dot(X.T, y) / (np.sqrt(np.sum(X ** 2, axis=0)) * np.sqrt(np.sum(y ** 2)))
    return corr

def f_regression(X, y, center=True, force_finite=True):
    """Perform univariate linear regression tests and return F-statistic and p-values."""
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Calculate correlation coefficients
    corr = r_regression(X, y, center=center)
    
    # Convert correlation coefficients to F-statistics
    n_samples = X.shape[0]
    degrees_of_freedom = n_samples - 2
    f_statistic = (corr ** 2) * degrees_of_freedom / (1 - corr ** 2)
    
    # Convert F-statistics to p-values
    p_values = stats.f.sf(f_statistic, 1, degrees_of_freedom)
    
    if force_finite:
        # Handle non-finite F-statistics
        f_statistic = np.where(np.isfinite(f_statistic), f_statistic, 0)
        p_values = np.where(np.isfinite(p_values), p_values, 1)
    
    return f_statistic, p_values

