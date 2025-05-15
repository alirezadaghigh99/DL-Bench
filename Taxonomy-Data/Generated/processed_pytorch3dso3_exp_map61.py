import torch

def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Converts a batch of logarithmic representations of rotation matrices to a batch of 3x3 rotation matrices.
    
    Args:
        log_rot (torch.Tensor): A batch of vectors of shape (minibatch, 3) representing the logarithmic representation of rotations.
        eps (float): A small value to handle the singularity around log(R) = 0.
        
    Returns:
        torch.Tensor: A batch of rotation matrices of shape (minibatch, 3, 3).
        
    Raises:
        ValueError: If the input log_rot is not of shape (minibatch, 3).
    """
    if log_rot.ndim != 2 or log_rot.shape[1] != 3:
        raise ValueError("Input log_rot must have shape (minibatch, 3)")

    # Compute the angle (theta) and the axis (k) from the log_rot
    theta = torch.norm(log_rot, dim=1, keepdim=True)  # Shape: (minibatch, 1)
    k = log_rot / (theta + eps)  # Normalize the axis, Shape: (minibatch, 3)

    # Handle the singularity by clamping theta
    theta = theta.clamp(min=eps)

    # Compute the skew-symmetric cross-product matrix of k
    K = torch.zeros((log_rot.size(0), 3, 3), device=log_rot.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    # Compute the rotation matrices using Rodrigues' formula
    I = torch.eye(3, device=log_rot.device).unsqueeze(0)  # Shape: (1, 3, 3)
    theta = theta.unsqueeze(-1)  # Shape: (minibatch, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.bmm(K, K)

    return R

