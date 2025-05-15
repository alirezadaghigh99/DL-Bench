import numpy as np

def hinge_loss(y_true, pred_decision, labels=None, sample_weight=None):
    y_true = np.asarray(y_true)
    pred_decision = np.asarray(pred_decision)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError("sample_weight and y_true must have the same length.")
    
    unique_labels = np.unique(y_true)
    
    if labels is not None:
        labels = np.asarray(labels)
    else:
        labels = unique_labels
    
    if len(unique_labels) > 2:
        # Multiclass hinge loss
        if pred_decision.ndim != 2 or pred_decision.shape[1] != len(labels):
            raise ValueError("pred_decision must be a 2D array with shape (n_samples, n_classes).")
        
        # Create a mask for the correct class
        correct_class_scores = pred_decision[np.arange(len(y_true)), y_true]
        
        # Calculate the hinge loss for each class
        margins = np.maximum(0, 1 + pred_decision - correct_class_scores[:, np.newaxis])
        margins[np.arange(len(y_true)), y_true] = 0  # Do not consider the correct class in the sum
        
        # Sum over classes and average
        loss = np.sum(margins, axis=1)
    else:
        # Binary hinge loss
        if pred_decision.ndim != 1:
            raise ValueError("pred_decision must be a 1D array for binary classification.")
        
        # Calculate the hinge loss
        loss = np.maximum(0, 1 - y_true * pred_decision)
    
    if sample_weight is not None:
        loss = loss * sample_weight
    
    return np.mean(loss)

