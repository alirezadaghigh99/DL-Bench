import torch

def quaternion_to_axis_angle(quaternion):
    # Check if the input is a tensor
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check if the input has the correct shape
    if quaternion.dim() == 1:
        if quaternion.size(0) != 4:
            raise ValueError("Input tensor must have shape [4] or [N, 4].")
    elif quaternion.dim() == 2:
        if quaternion.size(1) != 4:
            raise ValueError("Input tensor must have shape [4] or [N, 4].")
    else:
        raise ValueError("Input tensor must have shape [4] or [N, 4].")
    
    # Normalize the quaternion to ensure it is a unit quaternion
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    
    # Unpack the quaternion components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Compute the angle of rotation
    angle = 2 * torch.acos(w)
    
    # Compute the axis of rotation
    sin_theta_over_two = torch.sqrt(1 - w**2)
    
    # To avoid division by zero, use a small epsilon
    epsilon = 1e-8
    axis = torch.stack((x, y, z), dim=-1) / (sin_theta_over_two.unsqueeze(-1) + epsilon)
    
    # If sin_theta_over_two is very small, the axis is not well-defined
    # In this case, we can set the axis to any arbitrary unit vector
    axis = torch.where(sin_theta_over_two.unsqueeze(-1) < epsilon, torch.tensor([1.0, 0.0, 0.0], device=quaternion.device), axis)
    
    # Combine the axis and angle into a single tensor
    axis_angle = axis * angle.unsqueeze(-1)
    
    return axis_angle

