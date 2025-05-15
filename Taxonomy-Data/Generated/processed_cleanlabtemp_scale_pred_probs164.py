import numpy as np

def temp_scale_pred_probs(pred_probs, temp):
    # Clip probabilities to avoid log(0)
    epsilon = 1e-12
    pred_probs = np.clip(pred_probs, epsilon, 1.0)
    
    # Normalize probabilities to ensure they sum to 1
    pred_probs /= np.sum(pred_probs, axis=1, keepdims=True)
    
    # Apply temperature scaling
    log_probs = np.log(pred_probs)
    scaled_log_probs = log_probs / temp
    
    # Apply softmax to get scaled probabilities
    exp_scaled_log_probs = np.exp(scaled_log_probs)
    scaled_probs = exp_scaled_log_probs / np.sum(exp_scaled_log_probs, axis=1, keepdims=True)
    
    return scaled_probs

