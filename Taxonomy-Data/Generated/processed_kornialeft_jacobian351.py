import torch

def skew_symmetric(vec):
    """
    Compute the skew-symmetric matrix of a vector.
    
    Args:
        vec: A tensor of shape (B, 3).
    
    Returns:
        A tensor of shape (B, 3, 3) representing the skew-symmetric matrices.
    """
    B = vec.shape[0]
    zero = torch.zeros(B, 1, 1, device=vec.device, dtype=vec.dtype)
    vec = vec.unsqueeze(-1)  # Shape (B, 3, 1)
    
    skew = torch.cat([
        zero, -vec[:, 2:3], vec[:, 1:2],
        vec[:, 2:3], zero, -vec[:, 0:1],
        -vec[:, 1:2], vec[:, 0:1], zero
    ], dim=1).reshape(B, 3, 3)
    
    return skew

def left_jacobian(vec):
    """
    Compute the left Jacobian of SO(3) for a batch of rotation vectors.
    
    Args:
        vec: A tensor of shape (B, 3) where B is the batch size.
    
    Returns:
        A tensor of shape (B, 3, 3) representing the left Jacobian matrices.
    """
    B = vec.shape[0]
    theta = vec.norm(dim=1, keepdim=True)  # Shape (B, 1)
    theta_sq = theta ** 2
    theta_cub = theta ** 3
    
    # Handle the case when theta is very small (use Taylor expansion)
    small_angle = theta < 1e-6
    large_angle = ~small_angle
    
    # Identity matrix
    I = torch.eye(3, device=vec.device, dtype=vec.dtype).unsqueeze(0).repeat(B, 1, 1)
    
    # Skew-symmetric matrix of vec
    phi_hat = skew_symmetric(vec)
    phi_hat_sq = torch.bmm(phi_hat, phi_hat)
    
    # Compute the left Jacobian
    J_l = I.clone()
    
    # For large angles
    if large_angle.any():
        cos_theta = torch.cos(theta[large_angle])
        sin_theta = torch.sin(theta[large_angle])
        
        J_l[large_angle] += ((1 - cos_theta) / theta_sq[large_angle]) * phi_hat[large_angle]
        J_l[large_angle] += ((theta[large_angle] - sin_theta) / theta_cub[large_angle]) * phi_hat_sq[large_angle]
    
    # For small angles, use Taylor expansion
    if small_angle.any():
        J_l[small_angle] += 0.5 * phi_hat[small_angle]
        J_l[small_angle] += (1/6) * phi_hat_sq[small_angle]
    
    return J_l

