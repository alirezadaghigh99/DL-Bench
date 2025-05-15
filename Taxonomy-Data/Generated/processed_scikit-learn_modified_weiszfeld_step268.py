import numpy as np

def _modified_weiszfeld_step(X, x_old):
    """
    Perform one iteration step of the modified Weiszfeld algorithm to approximate the spatial median.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data points.
    x_old : ndarray, shape (n_features,)
        The current estimate of the spatial median.

    Returns:
    x_new : ndarray, shape (n_features,)
        The updated estimate of the spatial median.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # Calculate the distances from the current estimate to each point
    distances = np.linalg.norm(X - x_old, axis=1)

    # Handle the case where the distance is zero to avoid division by zero
    # Use a small epsilon to avoid numerical instability
    epsilon = 1e-10
    distances = np.where(distances < epsilon, epsilon, distances)

    # Calculate the weights as the inverse of the distances
    weights = 1.0 / distances

    # Compute the weighted sum of the data points
    weighted_sum = np.sum(weights[:, np.newaxis] * X, axis=0)

    # Compute the sum of the weights
    sum_weights = np.sum(weights)

    # Calculate the new estimate of the spatial median
    x_new = weighted_sum / sum_weights

    return x_new