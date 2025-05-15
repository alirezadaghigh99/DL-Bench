from sklearn.metrics import pair_confusion_matrix
import numpy as np

def rand_score(labels_true, labels_pred):
    """
    Calculate the Rand Index (RI) between two clusterings.

    Parameters:
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns:
    float
        The Rand Index, a similarity measure between 0.0 and 1.0.
    """
    # Calculate the pair confusion matrix
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    
    # Extract the elements of the pair confusion matrix
    tn, fp, fn, tp = contingency.ravel()
    
    # Calculate the numerator and denominator for the Rand Index
    numerator = tp + tn
    denominator = tp + tn + fp + fn
    
    # Handle special cases
    if denominator == 0:
        return 1.0
    if numerator == denominator:
        return 1.0
    
    # Calculate the Rand Index
    ri = numerator / denominator
    return ri

