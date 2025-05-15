import torch
from typing import Optional

class Transform3d:
    # Assuming Transform3d is a base class with necessary methods and properties
    pass

class Rotate(Transform3d):
    def __init__(
        self,
        R: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        # Ensure R is a tensor and move it to the specified device
        R = R.to(dtype=dtype, device=device)
        
        # Check the shape of R
        if R.ndim == 2:
            if R.shape != (3, 3):
                raise ValueError("R must have shape (3, 3) or (N, 3, 3)")
            R = R.unsqueeze(0)  # Add batch dimension for consistency
        elif R.ndim == 3:
            if R.shape[1:] != (3, 3):
                raise ValueError("R must have shape (3, 3) or (N, 3, 3)")
        else:
            raise ValueError("R must have shape (3, 3) or (N, 3, 3)")

        # Check orthogonality and determinant
        batch_size = R.shape[0]
        identity = torch.eye(3, dtype=dtype, device=device).expand(batch_size, -1, -1)
        R_transpose = R.transpose(1, 2)
        should_be_identity = torch.bmm(R, R_transpose)
        if not torch.allclose(should_be_identity, identity, atol=orthogonal_tol):
            raise ValueError("R is not orthogonal within the specified tolerance")

        det_R = torch.det(R)
        if not torch.allclose(det_R, torch.ones(batch_size, dtype=dtype, device=device), atol=orthogonal_tol):
            raise ValueError("Determinant of R is not 1 within the specified tolerance")

        # Store the rotation matrix
        self.R = R
        self.dtype = dtype
        self.device = device
        self.orthogonal_tol = orthogonal_tol

        # Call the parent class constructor if needed
        super().__init__()

