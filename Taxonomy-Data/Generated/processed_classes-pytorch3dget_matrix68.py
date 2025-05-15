import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self.device = device
        self.dtype = dtype
        self._transforms = []
        
        if matrix is not None:
            self._matrix = matrix.to(dtype=dtype, device=device)
        else:
            # Initialize with an identity matrix if no matrix is provided
            self._matrix = torch.eye(4, dtype=dtype, device=device)
    
    def add_transform(self, transform_matrix: torch.Tensor):
        """Add a transformation matrix to the list of transforms."""
        self._transforms.append(transform_matrix.to(dtype=self.dtype, device=self.device))
    
    def get_matrix(self) -> torch.Tensor:
        # Start with the initial matrix
        result_matrix = self._matrix
        
        # Iterate through each transform and multiply the matrices
        for transform in self._transforms:
            # Ensure the transform is a 4x4 matrix
            if transform.shape[-2:] != (4, 4):
                raise ValueError("Each transform must be a 4x4 matrix.")
            
            # Multiply the current result matrix with the next transform
            result_matrix = torch.matmul(result_matrix, transform)
        
        return result_matrix

