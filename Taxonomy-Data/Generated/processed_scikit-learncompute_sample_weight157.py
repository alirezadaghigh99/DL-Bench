import numpy as np
from collections import Counter

def compute_sample_weight(class_weight, y, indices=None):
    # Convert y to a numpy array if it isn't already
    y = np.asarray(y)
    
    # If indices are provided, use them to create a subsample of y
    if indices is not None:
        y = y[indices]
    
    # Get the unique classes and their counts
    classes, class_counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    
    # Initialize sample weights to ones
    sample_weight_vect = np.ones(n_samples, dtype=np.float64)
    
    # Handle different types of class_weight
    if class_weight is None:
        # No weighting, return the default sample weights
        return sample_weight_vect
    
    elif class_weight == "balanced":
        # Calculate balanced weights: n_samples / (n_classes * class_counts)
        class_weight_dict = {cls: n_samples / (len(classes) * count) for cls, count in zip(classes, class_counts)}
    
    elif isinstance(class_weight, dict):
        # Use the provided class_weight dictionary
        class_weight_dict = class_weight
    
    elif isinstance(class_weight, list):
        # If a list of dicts is provided, ensure it matches the number of classes
        if len(class_weight) != len(classes):
            raise ValueError("Length of class_weight list must match number of classes.")
        class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weight)}
    
    else:
        raise ValueError("class_weight must be 'balanced', a dict, a list of dicts, or None.")
    
    # Apply the class weights to the sample weights
    for cls, weight in class_weight_dict.items():
        sample_weight_vect[y == cls] = weight
    
    return sample_weight_vect

