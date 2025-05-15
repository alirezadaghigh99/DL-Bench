import torch

def vector_to_skew_symmetric_matrix(vec):
    """
    Convert a vector of shape (B, 3) to a skew-symmetric matrix of shape (B, 3, 3).
    """
    B = vec.shape[0]
    skew_matrices = torch.zeros((B, 3, 3), dtype=vec.dtype, device=vec.device)
    skew_matrices[:, 0, 1] = -vec[:, 2]
    skew_matrices[:, 0, 2] = vec[:, 1]
    skew_matrices[:, 1, 0] = vec[:, 2]
    skew_matrices[:, 1, 2] = -vec[:, 0]
    skew_matrices[:, 2, 0] = -vec[:, 1]
    skew_matrices[:, 2, 1] = vec[:, 0]
    return skew_matrices

def right_jacobian(vec):
    """
    Compute the right Jacobian of SO(3) for a batch of vectors.
    
    Parameters:
    vec (torch.Tensor): A tensor of shape (B, 3) representing the input vectors.
    
    Returns:
    torch.Tensor: A tensor of shape (B, 3, 3) representing the right Jacobian matrices.
    """
    B = vec.shape[0]
    norm_vec = torch.norm(vec, dim=1, keepdim=True)  # Shape: (B, 1)
    norm_vec = norm_vec.clamp(min=1e-8)  # Avoid division by zero

    # Identity matrix of shape (3, 3)
    I = torch.eye(3, dtype=vec.dtype, device=vec.device).unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, 3, 3)

    # Skew-symmetric matrix of the input vector
    skew_vec = vector_to_skew_symmetric_matrix(vec)  # Shape: (B, 3, 3)

    # Compute the right Jacobian
    theta = norm_vec.squeeze(-1)  # Shape: (B,)
    theta2 = theta ** 2
    theta3 = theta ** 3

    # Use the series expansion for small angles
    term1 = (1 - torch.cos(theta)) / theta2.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
    term2 = (theta - torch.sin(theta)) / theta3.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)

    J_right = I - term1 * skew_vec + term2 * torch.bmm(skew_vec, skew_vec)  # Shape: (B, 3, 3)

    return J_right

