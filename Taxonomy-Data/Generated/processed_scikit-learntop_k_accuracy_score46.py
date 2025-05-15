import numpy as np

def top_k_accuracy_score(y_true, y_score, k=1, normalize=True, sample_weight=None, labels=None):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Handle binary classification case
    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T
    
    n_samples, n_classes = y_score.shape
    
    # Validate k
    if not (1 <= k <= n_classes):
        raise ValueError(f"k should be between 1 and {n_classes}, got {k}.")
    
    # Handle labels
    if labels is not None:
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        y_true = np.array([label_to_index[label] for label in y_true])
    
    # Get the indices of the top-k scores
    top_k_indices = np.argsort(y_score, axis=1)[:, -k:]
    
    # Check if true labels are in the top-k indices
    correct = np.any(top_k_indices == y_true[:, None], axis=1)
    
    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        correct = correct * sample_weight
    
    # Calculate the score
    if normalize:
        return np.sum(correct) / np.sum(sample_weight) if sample_weight is not None else np.mean(correct)
    else:
        return np.sum(correct)

