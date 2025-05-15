import torch

def set_laf_orientation(LAF, angles_degrees):
    """
    Change the orientation of the Local Affine Frames (LAFs).

    Parameters:
    LAF (torch.Tensor): A tensor of shape (B, N, 2, 3) representing the LAFs.
    angles_degrees (torch.Tensor): A tensor of shape (B, N, 1) representing the angles in degrees.

    Returns:
    torch.Tensor: A tensor of shape (B, N, 2, 3) representing the LAFs oriented with the specified angles.
    """
    # Convert angles from degrees to radians
    angles_radians = angles_degrees * (torch.pi / 180.0)

    # Compute the rotation matrices
    cos_angles = torch.cos(angles_radians)
    sin_angles = torch.sin(angles_radians)

    # Create rotation matrices of shape (B, N, 2, 2)
    rotation_matrices = torch.zeros(LAF.shape[0], LAF.shape[1], 2, 2, device=LAF.device)
    rotation_matrices[:, :, 0, 0] = cos_angles.squeeze(-1)
    rotation_matrices[:, :, 0, 1] = -sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 0] = sin_angles.squeeze(-1)
    rotation_matrices[:, :, 1, 1] = cos_angles.squeeze(-1)

    # Extract the affine part of the LAF (first two columns)
    affine_part = LAF[:, :, :, :2]

    # Rotate the affine part
    rotated_affine_part = torch.matmul(rotation_matrices, affine_part)

    # Construct the new LAF with the rotated affine part and the original translation part
    new_LAF = torch.cat((rotated_affine_part, LAF[:, :, :, 2:]), dim=-1)

    return new_LAF