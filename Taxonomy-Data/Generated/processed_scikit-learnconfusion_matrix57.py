import numpy as np

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Determine the unique labels
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)
    
    # Create a mapping from label to index
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    # Initialize the confusion matrix
    n_labels = len(labels)
    C = np.zeros((n_labels, n_labels), dtype=np.int64)
    
    # Populate the confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            true_index = label_to_index[true]
            pred_index = label_to_index[pred]
            C[true_index, pred_index] += 1
    
    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true in label_to_index and pred in label_to_index:
                true_index = label_to_index[true]
                pred_index = label_to_index[pred]
                C[true_index, pred_index] += sample_weight[i] - 1
    
    # Normalize the confusion matrix if required
    if normalize is not None:
        with np.errstate(all='ignore'):
            if normalize == 'true':
                C = C / C.sum(axis=1, keepdims=True)
            elif normalize == 'pred':
                C = C / C.sum(axis=0, keepdims=True)
            elif normalize == 'all':
                C = C / C.sum()
            else:
                raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")
    
    # Handle any NaN values that may have resulted from division by zero
    C = np.nan_to_num(C)
    
    return C

