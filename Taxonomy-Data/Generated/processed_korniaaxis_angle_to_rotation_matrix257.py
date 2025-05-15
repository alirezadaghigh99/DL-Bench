import torch

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert 3D vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        axis_angle: tensor of 3D vector of axis-angle rotations in radians with shape (N, 3).

    Returns:
        tensor of rotation matrices of shape (N, 3, 3).
    """
    # Ensure the input is a tensor
    if not isinstance(axis_angle, torch.Tensor):
        axis_angle = torch.tensor(axis_angle, dtype=torch.float32)

    # Get the angle (magnitude of the vector)
    angles = torch.norm(axis_angle, dim=1, keepdim=True)

    # Avoid division by zero by setting zero angles to one (the result will be identity matrix)
    angles = angles + (angles == 0).float()

    # Normalize the axis
    axis = axis_angle / angles

    # Compute the components of the Rodrigues' rotation formula
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    one_minus_cos = 1 - cos_angles

    # Extract the components of the axis
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    # Compute the rotation matrix components
    R = torch.zeros((axis_angle.shape[0], 3, 3), dtype=torch.float32)
    R[:, 0, 0] = cos_angles + x * x * one_minus_cos
    R[:, 0, 1] = x * y * one_minus_cos - z * sin_angles
    R[:, 0, 2] = x * z * one_minus_cos + y * sin_angles

    R[:, 1, 0] = y * x * one_minus_cos + z * sin_angles
    R[:, 1, 1] = cos_angles + y * y * one_minus_cos
    R[:, 1, 2] = y * z * one_minus_cos - x * sin_angles

    R[:, 2, 0] = z * x * one_minus_cos - y * sin_angles
    R[:, 2, 1] = z * y * one_minus_cos + x * sin_angles
    R[:, 2, 2] = cos_angles + z * z * one_minus_cos

    return R

