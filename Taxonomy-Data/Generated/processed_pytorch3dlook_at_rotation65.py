import torch
import numpy as np

def normalize(v):
    """Normalize a vector or a batch of vectors."""
    return v / torch.norm(v, dim=-1, keepdim=True)

def look_at_rotation(camera_position, at, up, device='cpu'):
    # Convert inputs to tensors if they are not already
    if not isinstance(camera_position, torch.Tensor):
        camera_position = torch.tensor(camera_position, dtype=torch.float32, device=device)
    if not isinstance(at, torch.Tensor):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, dtype=torch.float32, device=device)

    # Ensure inputs are 2D tensors
    if camera_position.dim() == 1:
        camera_position = camera_position.unsqueeze(0)
    if at.dim() == 1:
        at = at.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    # Calculate the forward (z) axis
    z_axis = normalize(at - camera_position)

    # Calculate the right (x) axis
    x_axis = torch.cross(normalize(up), z_axis)
    x_axis = normalize(x_axis)

    # Handle the case where the x-axis is close to zero
    # If x_axis is zero, it means up and z_axis are collinear, so we need a different up vector
    zero_mask = torch.norm(x_axis, dim=-1) < 1e-6
    if zero_mask.any():
        alternative_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(up)
        alternative_x_axis = torch.cross(alternative_up, z_axis)
        alternative_x_axis = normalize(alternative_x_axis)
        x_axis = torch.where(zero_mask.unsqueeze(-1), alternative_x_axis, x_axis)

    # Calculate the true up (y) axis
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)

    # Concatenate the axes to form the rotation matrix
    R = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    # Return the transposed matrix to transform world coordinates to view coordinates
    return R.transpose(-1, -2)

