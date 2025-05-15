import numpy as np
from numpy.typing import NDArray

class RCF:
    def __init__(self, in_channels: int = 4, features: int = 16, kernel_size: int = 3, bias: float = -1.0, seed: int | None = None, mode: str = 'gaussian', dataset: 'NonGeoDataset' | None = None):
        self.weights = ...
        self.biases = ...
        pass

    def _normalize(self, patches: NDArray[np.float32], min_divisor: float = 1e-8, zca_bias: float = 0.001) -> NDArray[np.float32]:
        # Reshape patches to (N, C*H*W) for easier manipulation
        N, C, H, W = patches.shape
        patches_reshaped = patches.reshape(N, -1)

        # Step 1: Remove the mean
        mean = np.mean(patches_reshaped, axis=0)
        patches_centered = patches_reshaped - mean

        # Step 2: Normalize to have unit norms
        norms = np.linalg.norm(patches_centered, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, min_divisor)
        patches_normalized = patches_centered / norms

        # Step 3: Compute the covariance matrix
        covariance_matrix = np.cov(patches_normalized, rowvar=False)

        # Step 4: Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(covariance_matrix)

        # Step 5: Compute the ZCA whitening matrix
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + zca_bias))
        zca_whitening_matrix = U @ S_inv_sqrt @ U.T

        # Step 6: Apply the ZCA whitening matrix
        patches_whitened = patches_normalized @ zca_whitening_matrix

        # Reshape back to the original shape (N, C, H, W)
        patches_whitened = patches_whitened.reshape(N, C, H, W)

        return patches_whitened