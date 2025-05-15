import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self._matrix = matrix if matrix is not None else torch.eye(4, dtype=dtype, device=device)
        self._transforms = []  # List to store individual transformations
        self.device = device
        self.dtype = dtype

    def compose(self) -> torch.Tensor:
        """Compose all stored transformations into a single matrix."""
        composed_matrix = self._matrix.clone()
        for transform in self._transforms:
            composed_matrix = transform @ composed_matrix
        return composed_matrix

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        if invert_composed:
            # Compose all transformations and then invert the result
            composed_matrix = self.compose()
            inverse_matrix = torch.linalg.inv(composed_matrix)
            return Transform3d(dtype=self.dtype, device=self.device, matrix=inverse_matrix)
        else:
            # Invert each transformation individually
            inverted_transforms = [torch.linalg.inv(transform) for transform in reversed(self._transforms)]
            inverse_transform = Transform3d(dtype=self.dtype, device=self.device)
            inverse_transform._transforms = inverted_transforms
            return inverse_transform

