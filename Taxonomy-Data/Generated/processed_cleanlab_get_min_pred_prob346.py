import numpy as np

def _get_min_pred_prob(predictions):
    # Initialize pred_probs with a value of 1.0
    pred_probs = [1.0]
    
    # Iterate through each prediction in the input list
    for prediction in predictions:
        # Extract the last column of each class prediction
        last_column = prediction[:, -1]
        # Append the extracted values to pred_probs
        pred_probs.extend(last_column)
    
    # Calculate the minimum value in the pred_probs list
    min_pred_prob = np.min(pred_probs)
    
    return min_pred_prob