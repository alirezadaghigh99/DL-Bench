import numpy as np
from sklearn.metrics import confusion_matrix

def _adapted_cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None):
    # Compute the confusion matrix
    cm = confusion_matrix(y1, y2, labels=labels, sample_weight=sample_weight)
    
    # Number of classes
    n_classes = len(cm)
    
    # Observed agreement
    observed_agreement = np.trace(cm)
    
    # Expected agreement
    sum0 = np.sum(cm, axis=0)
    sum1 = np.sum(cm, axis=1)
    expected_agreement = np.dot(sum0, sum1) / np.sum(cm)
    
    # Total number of samples
    total_samples = np.sum(cm)
    
    # Handle the special case of perfect agreement
    if observed_agreement == total_samples:
        return 1.0
    
    # Calculate kappa
    po = observed_agreement / total_samples
    pe = expected_agreement / total_samples
    
    # Prevent division by zero
    if pe == 1:
        return 0.0
    
    kappa = (po - pe) / (1 - pe)
    
    return kappa

