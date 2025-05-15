import numpy as np

def _apply_similarity_transform(X, R, T, s):
    """
    Applies a similarity transformation to a batch of point clouds.

    Parameters:
    - X: np.ndarray of shape (minibatch, num_points, d)
        The input point clouds.
    - R: np.ndarray of shape (minibatch, d, d)
        The batch of orthonormal rotation matrices.
    - T: np.ndarray of shape (minibatch, d)
        The batch of translation vectors.
    - s: np.ndarray of shape (minibatch,)
        The batch of scaling factors.

    Returns:
    - transformed_X: np.ndarray of shape (minibatch, num_points, d)
        The transformed point clouds.
    """
    # Ensure the input arrays have compatible shapes
    assert X.shape[0] == R.shape[0] == T.shape[0] == s.shape[0], "Batch sizes must match"
    assert X.shape[2] == R.shape[1] == R.shape[2] == T.shape[1], "Dimensionality must match"

    # Apply the similarity transformation
    # Scale and rotate the point clouds
    scaled_rotated_X = np.einsum('bij,bkj->bki', R, X) * s[:, np.newaxis, np.newaxis]

    # Translate the point clouds
    transformed_X = scaled_rotated_X + T[:, np.newaxis, :]

    return transformed_X