import torch
import math

def get_rotation_matrix2d(center: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Calculate the affine matrix of 2D rotation.

    Args:
        center (torch.Tensor): Center of the rotation in the source image with shape (B, 2).
        angle (torch.Tensor): Rotation angle in degrees with shape (B).
        scale (torch.Tensor): Scale factor for x, y scaling with shape (B, 2).

    Returns:
        torch.Tensor: The affine matrix of 2D rotation with shape (B, 2, 3).
    """
    assert center.shape[1] == 2, "Center should have shape (B, 2)"
    assert angle.dim() == 1, "Angle should have shape (B)"
    assert scale.shape[1] == 2, "Scale should have shape (B, 2)"
    assert center.shape[0] == angle.shape[0] == scale.shape[0], "Batch size should be the same for all inputs"

    B = center.shape[0]
    angle_rad = angle * math.pi / 180.0  # Convert angle from degrees to radians

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Create the rotation matrix
    rotation_matrix = torch.zeros((B, 2, 3), dtype=center.dtype, device=center.device)

    # Fill the rotation matrix
    rotation_matrix[:, 0, 0] = cos_a * scale[:, 0]
    rotation_matrix[:, 0, 1] = -sin_a * scale[:, 1]
    rotation_matrix[:, 1, 0] = sin_a * scale[:, 0]
    rotation_matrix[:, 1, 1] = cos_a * scale[:, 1]

    # Calculate the translation part to keep the center in place
    rotation_matrix[:, 0, 2] = center[:, 0] - rotation_matrix[:, 0, 0] * center[:, 0] - rotation_matrix[:, 0, 1] * center[:, 1]
    rotation_matrix[:, 1, 2] = center[:, 1] - rotation_matrix[:, 1, 0] * center[:, 0] - rotation_matrix[:, 1, 1] * center[:, 1]

    return rotation_matrix

