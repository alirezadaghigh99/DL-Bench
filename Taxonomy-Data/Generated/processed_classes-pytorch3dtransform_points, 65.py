import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self.device = device
        self.dtype = dtype
        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device)
        else:
            self._matrix = matrix.to(dtype=dtype, device=device)
        self._transforms = []

    def transform_points(self, points, eps: Optional[float] = None) -> torch.Tensor:
        # Check the shape of the input points
        if points.dim() == 2 and points.size(1) == 3:
            # (P, 3) format
            points = points.unsqueeze(0)  # Add batch dimension
            single_batch = True
        elif points.dim() == 3 and points.size(2) == 3:
            # (N, P, 3) format
            single_batch = False
        else:
            raise ValueError("Points should be of shape (P, 3) or (N, P, 3)")

        # Get the batch size and number of points
        N, P, _ = points.shape

        # Convert points to homogeneous coordinates (N, P, 4)
        ones = torch.ones((N, P, 1), dtype=points.dtype, device=points.device)
        points_homogeneous = torch.cat([points, ones], dim=-1)

        # Apply the transformation matrix
        # Reshape matrix for batch multiplication
        matrix = self._matrix.expand(N, -1, -1)
        transformed_points_homogeneous = torch.bmm(points_homogeneous, matrix.transpose(1, 2))

        # Normalize by the homogeneous coordinate
        w = transformed_points_homogeneous[..., 3:4]
        if eps is not None:
            w = torch.clamp(w, min=eps)
        transformed_points = transformed_points_homogeneous[..., :3] / w

        # If the input was a single batch, remove the batch dimension
        if single_batch:
            transformed_points = transformed_points.squeeze(0)

        return transformed_points