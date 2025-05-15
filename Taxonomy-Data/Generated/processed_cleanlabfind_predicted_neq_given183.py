import numpy as np

def _find_predicted_neq_given_multilabel(labels, pred_probs):
    """
    Helper function to identify label issues in a multi-label setting.
    
    Parameters:
    - labels: np.ndarray or list of shape (n_samples, n_classes)
    - pred_probs: np.ndarray of shape (n_samples, n_classes)
    
    Returns:
    - np.ndarray: Boolean mask where True indicates a label issue.
    """
    # Convert labels to a numpy array if it's a list
    labels = np.array(labels)
    
    # Threshold to determine predicted labels from probabilities
    threshold = 0.5
    
    # Predicted labels based on threshold
    predicted_labels = (pred_probs >= threshold)
    
    # Compare predicted labels with actual labels
    label_issues = (predicted_labels != labels)
    
    # Any mismatch in the multi-label setting is considered a label issue
    return np.any(label_issues, axis=1)

def find_predicted_neq_given(labels, pred_probs, multi_label=False):
    """
    Identify label issues in a dataset using a simple baseline approach.
    
    Parameters:
    - labels: np.ndarray or list of shape (n_samples,) or (n_samples, n_classes)
    - pred_probs: np.ndarray of shape (n_samples, n_classes) for multi-label
                  or (n_samples, n_classes) for single-label
    - multi_label: bool, optional, default=False. If True, handles multi-label classification.
    
    Returns:
    - np.ndarray: Boolean mask where True indicates a label issue.
    """
    # Input validation
    if not isinstance(labels, (np.ndarray, list)):
        raise ValueError("Labels must be a numpy array or a list.")
    
    if not isinstance(pred_probs, np.ndarray):
        raise ValueError("Predicted probabilities must be a numpy array.")
    
    labels = np.array(labels)
    
    if multi_label:
        if labels.ndim != 2 or pred_probs.ndim != 2:
            raise ValueError("For multi-label, both labels and pred_probs must be 2D arrays.")
        if labels.shape != pred_probs.shape:
            raise ValueError("Labels and pred_probs must have the same shape for multi-label.")
        return _find_predicted_neq_given_multilabel(labels, pred_probs)
    else:
        if labels.ndim != 1 or pred_probs.ndim != 2:
            raise ValueError("For single-label, labels must be 1D and pred_probs must be 2D.")
        if labels.shape[0] != pred_probs.shape[0]:
            raise ValueError("Number of samples in labels and pred_probs must match for single-label.")
        
        # Predicted class is the one with the highest probability
        predicted_labels = np.argmax(pred_probs, axis=1)
        
        # High confidence threshold
        high_confidence_threshold = 0.9
        
        # High confidence predictions
        high_confidence = np.max(pred_probs, axis=1) >= high_confidence_threshold
        
        # Label issues are where predicted labels do not match given labels
        label_issues = (predicted_labels != labels)
        
        # Return mask where True indicates a label issue with high confidence
        return label_issues & high_confidence

