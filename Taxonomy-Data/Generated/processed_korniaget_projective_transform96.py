import torch
import math

def get_projective_transform(center, angles, scales):
    # Check input shapes
    assert center.shape[-1] == 3, "Center must have shape (B, 3)"
    assert angles.shape[-1] == 3, "Angles must have shape (B, 3)"
    assert scales.shape[-1] == 3, "Scales must have shape (B, 3)"
    assert center.shape[0] == angles.shape[0] == scales.shape[0], "Center, angles, and scales must have the same batch size"
    assert center.device == angles.device == scales.device, "Center, angles, and scales must be on the same device"
    assert center.dtype == angles.dtype == scales.dtype, "Center, angles, and scales must have the same dtype"

    B = center.shape[0]
    dtype = center.dtype
    device = center.device

    # Convert angles from degrees to radians
    angles_rad = angles * (math.pi / 180.0)

    # Compute rotation matrices for each axis
    cos = torch.cos(angles_rad)
    sin = torch.sin(angles_rad)

    # Rotation matrix around x-axis
    Rx = torch.stack([
        torch.tensor([1, 0, 0], dtype=dtype, device=device).expand(B, -1),
        torch.stack([torch.zeros(B, dtype=dtype, device=device), cos[:, 0], -sin[:, 0]], dim=1),
        torch.stack([torch.zeros(B, dtype=dtype, device=device), sin[:, 0], cos[:, 0]], dim=1)
    ], dim=1)

    # Rotation matrix around y-axis
    Ry = torch.stack([
        torch.stack([cos[:, 1], torch.zeros(B, dtype=dtype, device=device), sin[:, 1]], dim=1),
        torch.tensor([0, 1, 0], dtype=dtype, device=device).expand(B, -1),
        torch.stack([-sin[:, 1], torch.zeros(B, dtype=dtype, device=device), cos[:, 1]], dim=1)
    ], dim=1)

    # Rotation matrix around z-axis
    Rz = torch.stack([
        torch.stack([cos[:, 2], -sin[:, 2], torch.zeros(B, dtype=dtype, device=device)], dim=1),
        torch.stack([sin[:, 2], cos[:, 2], torch.zeros(B, dtype=dtype, device=device)], dim=1),
        torch.tensor([0, 0, 1], dtype=dtype, device=device).expand(B, -1)
    ], dim=1)

    # Combined rotation matrix R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # Apply scaling
    S = torch.diag_embed(scales)

    # Combined rotation and scaling
    RS = torch.bmm(R, S)

    # Create the translation component
    T = center - torch.bmm(RS, center.unsqueeze(-1)).squeeze(-1)

    # Construct the projection matrix
    projection_matrix = torch.cat([RS, T.unsqueeze(-1)], dim=2)

    return projection_matrix

