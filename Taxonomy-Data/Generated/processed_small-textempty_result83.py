import numpy as np
from scipy.sparse import csr_matrix

def empty_result(multi_label, num_classes, return_prediction=True, return_proba=True):
    if not return_prediction and not return_proba:
        raise ValueError("Invalid usage: At least one of 'prediction' or 'proba' must be True")
    
    predictions = None
    proba = None
    
    if return_prediction:
        if multi_label:
            # For multi-label, use a sparse matrix for predictions
            predictions = csr_matrix((0, num_classes), dtype=np.int64)
        else:
            # For single-label, use an empty array for predictions
            predictions = np.empty((0,), dtype=np.int64)
    
    if return_proba:
        if multi_label:
            # For multi-label, use a sparse matrix for probabilities
            proba = csr_matrix((0, num_classes), dtype=float)
        else:
            # For single-label, use an empty 2D array for probabilities
            proba = np.empty((0, num_classes), dtype=float)
    
    if return_prediction and return_proba:
        return predictions, proba
    elif return_prediction:
        return predictions
    else:
        return proba

