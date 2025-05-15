import numpy as np
from scipy.sparse import csr_matrix

def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False, dtype=None):
    """
    Build a contingency matrix describing the relationship between ground truth class labels and cluster labels.

    Parameters:
    - labels_true: array-like of shape (n_samples,), Ground truth class labels.
    - labels_pred: array-like of shape (n_samples,), Cluster labels.
    - eps: float, optional, If not None, a small value added to all entries of the contingency matrix.
    - sparse: bool, optional, If True, return a sparse CSR matrix.
    - dtype: numeric type, optional, Desired data type of the output matrix.

    Returns:
    - C: ndarray or sparse matrix of shape (n_classes_true, n_classes_pred), Contingency matrix.

    Example:
    >>> labels_true = [0, 0, 1, 1, 2, 2]
    >>> labels_pred = [0, 0, 1, 1, 0, 2]
    >>> contingency_matrix(labels_true, labels_pred)
    array([[2, 0, 0],
           [0, 2, 0],
           [1, 0, 1]])
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    classes_true, class_idx_true = np.unique(labels_true, return_inverse=True)
    classes_pred, class_idx_pred = np.unique(labels_pred, return_inverse=True)

    n_classes_true = classes_true.shape[0]
    n_classes_pred = classes_pred.shape[0]

    # Initialize the contingency matrix
    if dtype is None:
        dtype = float if eps is not None else int

    C = np.zeros((n_classes_true, n_classes_pred), dtype=dtype)

    # Populate the contingency matrix
    for i in range(labels_true.size):
        C[class_idx_true[i], class_idx_pred[i]] += 1

    # Add epsilon if specified
    if eps is not None:
        C += eps

    # Convert to sparse matrix if required
    if sparse:
        C = csr_matrix(C)

    return C

