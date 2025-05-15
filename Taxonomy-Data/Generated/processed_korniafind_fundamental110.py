import numpy as np
import cv2

def find_fundamental(points1, points2, weights, method='8POINT'):
    """
    Compute the fundamental matrix using the specified method.

    Args:
        points1: A set of points in the first image with a tensor shape (B, N, 2), N>=8.
        points2: A set of points in the second image with a tensor shape (B, N, 2), N>=8.
        weights: Tensor containing the weights per point correspondence with a shape of (B, N).
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        The computed fundamental matrix with shape (B, 3*m, 3), where `m` is the number of fundamental matrices.

    Raises:
        ValueError: If an invalid method is provided.
    """
    if method not in ['7POINT', '8POINT']:
        raise ValueError("Invalid method provided. Supported methods are '7POINT' and '8POINT'.")

    B, N, _ = points1.shape
    fundamental_matrices = []

    for b in range(B):
        pts1 = points1[b]
        pts2 = points2[b]
        w = weights[b]

        # Normalize points
        pts1 = pts1 * w[:, np.newaxis]
        pts2 = pts2 * w[:, np.newaxis]

        if method == '7POINT' and N >= 7:
            F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_7POINT)
        elif method == '8POINT' and N >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
        else:
            raise ValueError(f"Insufficient points for {method} method. Required: {7 if method == '7POINT' else 8}, Given: {N}")

        if F is not None:
            # Reshape F to (3*m, 3) where m is the number of solutions
            F = F.reshape(-1, 3, 3)
            fundamental_matrices.append(F)
        else:
            raise ValueError("Fundamental matrix computation failed.")

    # Stack all fundamental matrices for each batch
    return np.array(fundamental_matrices)

