import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        if matrix is None:
            # Default to an identity matrix if no matrix is provided
            self._matrix = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)  # Add batch dimension
        else:
            self._matrix = matrix.to(dtype=dtype, device=device)
        
        self.device = device
        self.dtype = dtype

    def stack(self, *others: "Transform3d") -> "Transform3d":
        # Collect all matrices to be stacked
        matrices = [self._matrix] + [other._matrix for other in others]
        
        # Check that all matrices have the same shape except for the batch dimension
        base_shape = matrices[0].shape[1:]  # Exclude batch dimension
        for matrix in matrices:
            if matrix.shape[1:] != base_shape:
                raise ValueError("All transformation matrices must have the same shape except for the batch dimension.")
        
        # Check that all matrices are on the same device and have the same dtype
        for matrix in matrices:
            if matrix.device != self.device or matrix.dtype != self.dtype:
                raise ValueError("All transformation matrices must be on the same device and have the same dtype.")
        
        # Concatenate matrices along the batch dimension
        stacked_matrix = torch.cat(matrices, dim=0)
        
        # Return a new Transform3d object with the stacked matrix
        return Transform3d(dtype=self.dtype, device=self.device, matrix=stacked_matrix)

