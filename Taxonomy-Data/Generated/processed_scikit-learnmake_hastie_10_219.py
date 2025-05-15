import numpy as np

def make_hastie_10_2(n_samples, random_state=None):
    """
    Generate data for binary classification as used in Hastie et al. 2009, Example 10.2.

    Parameters:
    - n_samples: int, number of samples to generate.
    - random_state: int or None, random seed for reproducibility.

    Returns:
    - X: ndarray of shape (n_samples, 10), input samples with standard independent Gaussian features.
    - y: ndarray of shape (n_samples,), output values where y[i] is 1 if the sum of X[i] squared is greater than 9.34, otherwise -1.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate standard independent Gaussian features
    X = np.random.randn(n_samples, 10)
    
    # Calculate the sum of squares for each sample
    sum_of_squares = np.sum(X**2, axis=1)
    
    # Define the target variable based on the condition
    y = np.where(sum_of_squares > 9.34, 1, -1)
    
    return X, y

