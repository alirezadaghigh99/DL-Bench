import torch

def quaternion_exp_to_log(quaternion, eps=1e-6):
    # Check if input is a tensor
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check if the shape is (*, 4)
    if quaternion.shape[-1] != 4:
        raise ValueError("Input tensor must have shape (*, 4).")
    
    # Unpack the quaternion into scalar (w) and vector (x, y, z) parts
    w = quaternion[..., 0]
    xyz = quaternion[..., 1:]
    
    # Compute the norm of the vector part
    norm_xyz = torch.norm(xyz, dim=-1)
    
    # Clamp the norm to avoid division by zero
    norm_xyz_clamped = torch.clamp(norm_xyz, min=eps)
    
    # Compute the angle
    angle = torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    
    # Compute the log map
    log_map = (angle / norm_xyz_clamped).unsqueeze(-1) * xyz
    
    # Handle the case where the norm is zero (pure real quaternion)
    log_map = torch.where(norm_xyz.unsqueeze(-1) < eps, torch.zeros_like(xyz), log_map)
    
    return log_map

