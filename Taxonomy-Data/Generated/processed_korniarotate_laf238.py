import torch

def rotate_laf(LAF, angles_degrees):
    # Check if the input LAF has the correct shape
    if LAF.ndim != 4 or LAF.shape[2:] != (2, 3):
        raise ValueError("Input LAF must have shape (B, N, 2, 3)")
    
    # Check if angles_degrees has the correct shape
    if angles_degrees.ndim != 3 or angles_degrees.shape[2] != 1:
        raise ValueError("angles_degrees must have shape (B, N, 1)")
    
    # Convert angles from degrees to radians
    angles_radians = angles_degrees * (torch.pi / 180.0)
    
    # Compute the rotation matrices
    cos_theta = torch.cos(angles_radians)
    sin_theta = torch.sin(angles_radians)
    
    # Create the rotation matrix for each angle
    rotation_matrices = torch.zeros((LAF.shape[0], LAF.shape[1], 2, 2), device=LAF.device, dtype=LAF.dtype)
    rotation_matrices[:, :, 0, 0] = cos_theta.squeeze(-1)
    rotation_matrices[:, :, 0, 1] = -sin_theta.squeeze(-1)
    rotation_matrices[:, :, 1, 0] = sin_theta.squeeze(-1)
    rotation_matrices[:, :, 1, 1] = cos_theta.squeeze(-1)
    
    # Apply the rotation to the LAF
    # LAF[:, :, :2, :2] is the affine part of the LAF
    rotated_affine_part = torch.matmul(rotation_matrices, LAF[:, :, :2, :2])
    
    # The translation part of the LAF remains unchanged
    translation_part = LAF[:, :, :, 2:]
    
    # Combine the rotated affine part with the unchanged translation part
    rotated_LAF = torch.cat((rotated_affine_part, translation_part), dim=-1)
    
    return rotated_LAF

