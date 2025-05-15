import numpy as np
from scipy.sparse import issparse
from scipy.stats import chi2 as chi2_dist
from sklearn.preprocessing import LabelBinarizer

def _chisquare(observed, expected):
    """Compute chi-squared statistic and p-value."""
    chi2_stat = ((observed - expected) ** 2 / expected).sum(axis=0)
    p_values = chi2_dist.sf(chi2_stat, df=observed.shape[0] - 1)
    return chi2_stat, p_values

def chi2(X, y):
    """Compute chi-squared statistics and p-values for each feature."""
    # Binarize the class labels
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    
    # Ensure X is in dense format if it's sparse
    if issparse(X):
        X = X.toarray()
    
    # Calculate observed frequencies
    observed = np.dot(Y.T, X)
    
    # Calculate expected frequencies
    feature_sums = X.sum(axis=0)
    class_prob = Y.mean(axis=0)
    expected = np.outer(class_prob, feature_sums)
    
    # Compute chi-squared statistics and p-values
    chi2_stat, p_values = _chisquare(observed, expected)
    
    return chi2_stat, p_values

