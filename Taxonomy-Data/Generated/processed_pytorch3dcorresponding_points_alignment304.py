import numpy as np
from collections import namedtuple

SimilarityTransform = namedtuple('SimilarityTransform', ['R', 'T', 's'])

def corresponding_points_alignment(X, Y, weights=None, estimate_scale=True, allow_reflection=False, eps=1e-8):
    # Ensure X and Y are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Check input dimensions
    if X.shape != Y.shape:
        raise ValueError("Point sets X and Y must have the same shape.")
    
    minibatch, num_points, d = X.shape
    
    if weights is None:
        weights = np.ones((minibatch, num_points))
    else:
        weights = np.asarray(weights)
        if weights.shape != (minibatch, num_points):
            raise ValueError("Weights should have the same first two dimensions as X.")
    
    # Normalize weights
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / (weights_sum + eps)
    
    # Compute weighted centroids
    mu_X = np.sum(weights[:, :, np.newaxis] * X, axis=1)
    mu_Y = np.sum(weights[:, :, np.newaxis] * Y, axis=1)
    
    # Center the points
    X_centered = X - mu_X[:, np.newaxis, :]
    Y_centered = Y - mu_Y[:, np.newaxis, :]
    
    # Compute covariance matrix
    cov_matrix = np.einsum('bij,bik,bjk->bik', weights[:, :, np.newaxis] * X_centered, Y_centered, np.eye(d))
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # Compute rotation matrix
    R = np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
    
    # Handle reflection case
    if not allow_reflection:
        det_R = np.linalg.det(R)
        Vt[det_R < 0, -1, :] *= -1
        R = np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
    
    # Compute scale
    if estimate_scale:
        var_X = np.sum(weights[:, :, np.newaxis] * (X_centered ** 2), axis=(1, 2))
        scale = np.sum(S, axis=1) / (var_X + eps)
    else:
        scale = np.ones(minibatch)
    
    # Compute translation
    T = mu_Y - scale[:, np.newaxis] * np.einsum('bij,bjk->bik', mu_X[:, np.newaxis, :], R).squeeze(1)
    
    return SimilarityTransform(R=R, T=T, s=scale)

