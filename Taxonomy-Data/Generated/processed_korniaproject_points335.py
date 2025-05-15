import torch

def project_points(points_3d, camera_matrix):
    """
    Projects 3D points onto a 2D camera plane using the camera matrix.

    Args:
        points_3d (torch.Tensor): A tensor of shape (*, 3) representing 3D points.
        camera_matrix (torch.Tensor): A tensor of shape (*, 3, 3) representing the camera matrix.

    Returns:
        torch.Tensor: A tensor of shape (*, 2) representing the projected 2D points.
    """
    # Ensure the points are in homogeneous coordinates by adding a dimension of ones
    ones = torch.ones(points_3d.shape[:-1] + (1,), dtype=points_3d.dtype, device=points_3d.device)
    points_3d_homogeneous = torch.cat((points_3d, ones), dim=-1)

    # Perform the matrix multiplication to project the points
    points_2d_homogeneous = torch.matmul(camera_matrix, points_3d_homogeneous.unsqueeze(-1)).squeeze(-1)

    # Convert from homogeneous coordinates to 2D coordinates
    u = points_2d_homogeneous[..., 0] / points_2d_homogeneous[..., 2]
    v = points_2d_homogeneous[..., 1] / points_2d_homogeneous[..., 2]

    # Stack the u and v coordinates to get the final 2D points
    points_2d = torch.stack((u, v), dim=-1)

    return points_2d

