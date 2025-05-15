import torch

def project_points_z1(points_in_camera):
    """
    Projects points from the camera frame into the canonical z=1 plane.

    Args:
        points_in_camera (torch.Tensor): A tensor of shape (..., 3) representing the points to project.

    Returns:
        torch.Tensor: A tensor of shape (..., 2) representing the projected points.
    """
    # Ensure the input is a tensor
    if not isinstance(points_in_camera, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Check that the last dimension is 3
    if points_in_camera.shape[-1] != 3:
        raise ValueError("The last dimension of points_in_camera must be 3")

    # Extract x, y, z components
    x = points_in_camera[..., 0]
    y = points_in_camera[..., 1]
    z = points_in_camera[..., 2]

    # Perform perspective division
    x_proj = x / z
    y_proj = y / z

    # Stack the results to get the final projected points
    projected_points = torch.stack((x_proj, y_proj), dim=-1)

    return projected_points

