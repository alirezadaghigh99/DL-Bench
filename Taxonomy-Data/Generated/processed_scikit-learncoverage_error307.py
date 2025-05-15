import numpy as np

def coverage_error(y_true, y_score, sample_weight=None):
    """
    Compute the coverage error measure.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        True binary labels in binary indicator format.
    
    y_score : array-like of shape (n_samples, n_labels)
        Target scores.
    
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    coverage_error : float
        The coverage error measure.
    
    User Guide
    ----------
    The coverage error measure calculates how far we need to go through the ranked scores
    to cover all true labels. It handles ties in y_scores by giving the maximal rank that
    would have been assigned to all tied values.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0])
    else:
        sample_weight = np.array(sample_weight)
    
    # Validate input dimensions
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    
    if y_true.shape[0] != sample_weight.shape[0]:
        raise ValueError("sample_weight must have the same number of samples as y_true and y_score.")
    
    n_samples, n_labels = y_true.shape
    coverage_errors = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Get the indices that would sort the scores in descending order
        sorted_indices = np.argsort(-y_score[i])
        
        # Find the rank of the highest true label
        max_rank = 0
        for j in range(n_labels):
            if y_true[i, sorted_indices[j]] == 1:
                max_rank = j + 1
        
        coverage_errors[i] = max_rank
    
    # Calculate the weighted average of the coverage errors
    weighted_coverage_error = np.average(coverage_errors, weights=sample_weight)
    
    return weighted_coverage_error

