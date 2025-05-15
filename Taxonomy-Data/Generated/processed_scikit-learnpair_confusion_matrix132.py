import numpy as np
from sklearn.metrics import pair_confusion_matrix as sklearn_pair_confusion_matrix

def pair_confusion_matrix(labels_true, labels_pred):
    """
    Calculate the 2x2 pair confusion matrix between two clusterings.

    Parameters:
    labels_true (array-like): Ground truth class labels.
    labels_pred (array-like): Cluster labels to evaluate.

    Returns:
    numpy.ndarray: A 2x2 contingency matrix.
    """
    # Use sklearn's pair_confusion_matrix to compute the matrix
    return sklearn_pair_confusion_matrix(labels_true, labels_pred)

