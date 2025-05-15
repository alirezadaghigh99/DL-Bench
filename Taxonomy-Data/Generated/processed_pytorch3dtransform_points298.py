import torch

class PointTransformer:
    def __init__(self, transformation_matrix):
        """
        Initialize the PointTransformer with a given transformation matrix.
        
        Args:
        - transformation_matrix (torch.Tensor): A tensor of shape (4, 4) representing the transformation matrix.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be of shape (4, 4).")
        self.transformation_matrix = transformation_matrix

    def get_matrix(self):
        """
        Returns the transformation matrix.
        
        Returns:
        - torch.Tensor: The transformation matrix of shape (4, 4).
        """
        return self.transformation_matrix

    def transform_points(self, points, eps=None):
        """
        Transforms a set of 3D points using the transformation matrix.
        
        Args:
        - points (torch.Tensor): A tensor of shape (P, 3) or (N, P, 3).
        - eps (float, optional): A small value to clamp the homogeneous coordinate to avoid division by zero.
        
        Returns:
        - torch.Tensor: The transformed points, in the same shape as the input.
        """
        # Validate input dimensions
        if points.dim() not in [2, 3]:
            raise ValueError("Input points tensor must be 2D or 3D.")

        # Reshape if necessary
        original_shape = points.shape
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Shape (1, P, 3)

        # Augment points with a column of ones
        ones = torch.ones(points.shape[:-1] + (1,), device=points.device, dtype=points.dtype)
        points_homogeneous = torch.cat([points, ones], dim=-1)  # Shape (N, P, 4)

        # Apply transformation
        transformed_points_homogeneous = torch.matmul(points_homogeneous, self.get_matrix().T)  # Shape (N, P, 4)

        # Extract the homogeneous coordinate
        w = transformed_points_homogeneous[..., 3]

        # Clamp if eps is provided
        if eps is not None:
            w = torch.clamp(w, min=eps)

        # Divide by the homogeneous coordinate
        transformed_points = transformed_points_homogeneous[..., :3] / w.unsqueeze(-1)

        # Reshape back if necessary
        if original_shape[0] == 1:
            transformed_points = transformed_points.squeeze(0)  # Shape (P, 3) if input was (P, 3)

        return transformed_points

