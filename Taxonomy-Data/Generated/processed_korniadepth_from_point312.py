import numpy as np

def depth_from_point(R, t, X):
    """
    Return the depth of a point transformed by a rigid transform.

    Args:
       R: The rotation matrix with shape (*, 3, 3).
       t: The translation vector with shape (*, 3, 1).
       X: The 3d points with shape (*, 3).

    Returns:
       The depth value per point with shape (*, 1).
    """
    # Ensure X is a column vector
    X = np.expand_dims(X, axis=-1)  # Shape becomes (*, 3, 1)

    # Apply the rigid transformation
    X_transformed = np.matmul(R, X) + t  # Shape (*, 3, 1)

    # Extract the z-component (depth)
    depth = X_transformed[..., 2, :]  # Shape (*, 1)

    return depth

