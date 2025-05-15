import numpy as np

def _bald(p, eps=1e-8):
    """
    Calculate the Bayesian Active Learning by Disagreement (BALD) score.

    Parameters:
    - p: A 2D numpy array of shape (n_samples, n_classes) representing the predicted probabilities
         for each class for each sample.
    - eps: A small value to avoid numerical instability in logarithms.

    Returns:
    - bald_scores: A 1D numpy array of shape (n_samples,) representing the BALD score for each sample.
    """
    # Ensure the input is a numpy array
    p = np.asarray(p)
    
    # Calculate the expected entropy
    expected_entropy = -np.sum(p * np.log(p + eps), axis=1)
    
    # Calculate the entropy of the expected probabilities
    mean_p = np.mean(p, axis=0)
    entropy_of_mean = -np.sum(mean_p * np.log(mean_p + eps))
    
    # Calculate the BALD score
    bald_scores = entropy_of_mean - expected_entropy
    
    return bald_scores

