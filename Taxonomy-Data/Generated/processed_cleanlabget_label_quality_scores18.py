import numpy as np

def get_label_quality_scores(labels, predictions, method='outre'):
    # Ensure inputs are numpy arrays
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    # Validate inputs
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    
    if method == 'outre':
        # Calculate the absolute differences between labels and predictions
        differences = np.abs(labels - predictions)
        
        # Normalize differences to a range between 0 and 1
        max_diff = np.max(differences)
        if max_diff == 0:
            # If max_diff is 0, it means all predictions are perfect
            return np.ones_like(differences)
        
        normalized_differences = differences / max_diff
        
        # Calculate quality scores as 1 - normalized differences
        quality_scores = 1 - normalized_differences
        
        return quality_scores
    else:
        raise ValueError(f"Unknown method: {method}")

