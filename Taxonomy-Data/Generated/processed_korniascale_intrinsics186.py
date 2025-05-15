import torch

def scale_intrinsics(camera_matrix, scale_factor):
    """
    Scales the focal length and center of projection in the camera matrix by the given scale factor.

    Parameters:
    - camera_matrix: A tensor of shape (B, 3, 3) containing the intrinsic parameters.
    - scale_factor: A float or a tensor that represents the scale factor.

    Returns:
    - A tensor of shape (B, 3, 3) with the scaled intrinsic parameters.
    """
    # Ensure the scale factor is a tensor if it's a float
    if isinstance(scale_factor, float):
        scale_factor = torch.tensor(scale_factor, dtype=camera_matrix.dtype, device=camera_matrix.device)

    # Scale the focal lengths and center of projection
    scaled_camera_matrix = camera_matrix.clone()
    scaled_camera_matrix[:, 0, 0] *= scale_factor  # Scale fx
    scaled_camera_matrix[:, 1, 1] *= scale_factor  # Scale fy
    scaled_camera_matrix[:, 0, 2] *= scale_factor  # Scale cx
    scaled_camera_matrix[:, 1, 2] *= scale_factor  # Scale cy

    return scaled_camera_matrix

