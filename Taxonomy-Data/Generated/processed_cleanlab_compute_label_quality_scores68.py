import numpy as np

def _compute_label_quality_scores(labels, predictions, method="objectlab", 
                                  aggregation_weights=None, threshold=None, 
                                  overlapping_label_check=True, verbose=True):
    if method != "objectlab":
        raise ValueError(f"Unsupported method: {method}. Only 'objectlab' is supported.")
    
    if len(labels) != len(predictions):
        raise ValueError("The number of labels must match the number of predictions.")
    
    scores = []
    
    for i, (label, prediction) in enumerate(zip(labels, predictions)):
        if verbose:
            print(f"Processing label-prediction pair {i+1}/{len(labels)}")
        
        