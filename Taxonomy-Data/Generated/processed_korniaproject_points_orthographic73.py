import torch

def project_points_orthographic(points_in_camera):
    """
    Project one or more points from the camera frame into the canonical z=1 plane through orthographic projection.

    Args:
        points_in_camera: Tensor representing the points to project. 
                          It should be of shape (..., 3), where the last dimension represents (x, y, z).

    Returns:
        Tensor representing the projected points, of shape (..., 2).
    """
    # Ensure the input is a tensor
    if not isinstance(points_in_camera, torch.Tensor):
        raise TypeError("points_in_camera must be a torch.Tensor")

    # Check if the last dimension is 3
    if points_in_camera.shape[-1] != 3:
        raise ValueError("The last dimension of points_in_camera must be 3, representing (x, y, z)")

    # Perform orthographic projection by selecting the first two components (x, y)
    projected_points = points_in_camera[..., :2]

    return projected_points

