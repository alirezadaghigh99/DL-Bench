import numpy as np

def _get_label_quality_per_image(pixel_scores, method='softmin', temperature=0.1):
    # Validate input
    if pixel_scores is None or len(pixel_scores) == 0:
        raise Exception("pixel_scores must be a non-empty NumPy array.")
    
    if temperature is None or temperature == 0:
        raise Exception("temperature must be a non-zero float.")
    
    if method != 'softmin':
        raise Exception("Unsupported method. Currently, only the 'softmin' method is supported.")
    
    # Calculate the softmin
    # Softmin is calculated as softmax(-x / temperature)
    scaled_scores = -pixel_scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
    softmin_scores = exp_scores / np.sum(exp_scores)
    
    # Calculate the label quality score
    label_quality_score = np.sum(softmin_scores * pixel_scores)
    
    return label_quality_score

