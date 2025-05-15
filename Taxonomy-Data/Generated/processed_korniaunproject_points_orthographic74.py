import torch

def unproject_points_orthographic(points_in_camera, extension):
    """
    Unprojects 2D points from the canonical z=1 plane into 3D space using the given extension.

    Args:
        points_in_camera (Tensor): A tensor of shape (..., 2) representing the 2D points.
        extension (Tensor): A tensor of shape (..., 1) representing the extension for each point.

    Returns:
        Tensor: A tensor of shape (..., 3) representing the unprojected 3D points.
    """
    # Concatenate the 2D points with the extension along the last dimension
    unprojected_points = torch.cat((points_in_camera, extension), dim=-1)
    return unprojected_points

