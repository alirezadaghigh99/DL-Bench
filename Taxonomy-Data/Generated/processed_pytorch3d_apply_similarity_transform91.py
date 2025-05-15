import numpy as np

def apply_similarity_transformation(X, R, s, T):
    """
    Applies a similarity transformation to a batch of point clouds.

    Parameters:
    - X: np.ndarray of shape (minibatch, num_points, d)
        The input batch of point clouds.
    - R: np.ndarray of shape (minibatch, d, d)
        The batch of orthonormal rotation matrices.
    - s: np.ndarray of shape (minibatch,)
        The batch of scaling factors.
    - T: np.ndarray of shape (minibatch, d)
        The batch of translation vectors.

    Returns:
    - transformed_X: np.ndarray of shape (minibatch, num_points, d)
        The transformed batch of point clouds.
    """
    # Ensure the input arrays have compatible shapes
    minibatch, num_points, d = X.shape
    assert R.shape == (minibatch, d, d), "R must have shape (minibatch, d, d)"
    assert s.shape == (minibatch,), "s must have shape (minibatch,)"
    assert T.shape == (minibatch, d), "T must have shape (minibatch, d)"

    # Apply the rotation
    rotated_X = np.einsum('bij,bpj->bpi', R, X)

    # Apply the scaling
    scaled_X = rotated_X * s[:, np.newaxis, np.newaxis]

    # Apply the translation
    transformed_X = scaled_X + T[:, np.newaxis, :]

    return transformed_X