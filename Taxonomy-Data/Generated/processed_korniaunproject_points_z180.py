import torch

def unproject_points_z1(points_in_cam_canonical, extension=None):
    """
    Unprojects points from the canonical z=1 plane into the camera frame.

    Args:
        points_in_cam_canonical (Tensor): A tensor of shape (..., 2) representing the points to unproject.
        extension (Tensor, optional): A tensor of shape (..., 1) representing the extension (depth) of the points.

    Returns:
        Tensor: A tensor of shape (..., 3) representing the unprojected points.
    """
    # Ensure points_in_cam_canonical is a tensor
    if not isinstance(points_in_cam_canonical, torch.Tensor):
        raise TypeError("points_in_cam_canonical must be a torch.Tensor")

    # Check the shape of points_in_cam_canonical
    if points_in_cam_canonical.shape[-1] != 2:
        raise ValueError("points_in_cam_canonical must have shape (..., 2)")

    # Create the z-coordinate tensor
    if extension is None:
        z = torch.ones_like(points_in_cam_canonical[..., :1])
    else:
        # Ensure extension is a tensor
        if not isinstance(extension, torch.Tensor):
            raise TypeError("extension must be a torch.Tensor")
        
        # Check the shape of extension
        if extension.shape != points_in_cam_canonical.shape[:-1] + (1,):
            raise ValueError("extension must have shape (..., 1) matching points_in_cam_canonical")

        z = extension

    # Concatenate the x, y coordinates with the z coordinate
    unprojected_points = torch.cat((points_in_cam_canonical, z), dim=-1)

    return unprojected_points

