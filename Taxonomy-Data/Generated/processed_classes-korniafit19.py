import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional

# Assuming zca_mean is a provided function
def zca_mean(x: Tensor, dim: int, unbiased: bool, eps: float, compute_inv: bool):
    # This is a placeholder for the actual implementation of zca_mean
    # It should return mean_vector, transform_matrix, and optionally transform_inv
    pass

class ZCAWhitening(Module):
    def __init__(
        self,
        dim: int = 0,
        eps: float = 1e-6,
        unbiased: bool = True,
        detach_transforms: bool = True,
        compute_inv: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.unbiased = unbiased
        self.detach_transforms = detach_transforms
        self.compute_inv = compute_inv
        self.fitted = False

        self.mean_vector: Optional[Tensor] = None
        self.transform_matrix: Optional[Tensor] = None
        self.transform_inv: Optional[Tensor] = None

    def fit(self, x: Tensor) -> None:
        # Compute the mean vector and transformation matrices
        mean_vector, transform_matrix, transform_inv = zca_mean(
            x, self.dim, self.unbiased, self.eps, self.compute_inv
        )

        # Detach the gradients if specified
        if self.detach_transforms:
            mean_vector = mean_vector.detach()
            transform_matrix = transform_matrix.detach()
            if transform_inv is not None:
                transform_inv = transform_inv.detach()

        # Assign the computed values to the class attributes
        self.mean_vector = mean_vector
        self.transform_matrix = transform_matrix
        self.transform_inv = transform_inv if transform_inv is not None else torch.empty(0)

        # Set the fitted flag to True
        self.fitted = True