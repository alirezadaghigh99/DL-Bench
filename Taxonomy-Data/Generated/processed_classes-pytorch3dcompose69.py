import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        if matrix is None:
            # Initialize with an identity matrix if no matrix is provided
            self._matrix = torch.eye(4, dtype=dtype, device=device)
        else:
            self._matrix = matrix.to(dtype=dtype, device=device)
        
        self._transforms = [self._matrix]
        self.device = device
        self.dtype = dtype

    def compose(self, *others: "Transform3d") -> "Transform3d":
        # Verify that all provided arguments are instances of Transform3d
        for other in others:
            if not isinstance(other, Transform3d):
                raise TypeError("All arguments must be instances of Transform3d")
        
        # Start with the current transformation matrix
        composed_matrix = self._matrix.clone()
        
        # Compose the matrices in the order they are provided
        for other in others:
            composed_matrix = composed_matrix @ other._matrix
        
        # Create a new Transform3d instance with the composed matrix
        new_transform = Transform3d(dtype=self.dtype, device=self.device, matrix=composed_matrix)
        
        # Update the internal list of transformations
        new_transform._transforms = self._transforms + [other._matrix for other in others]
        
        return new_transform

