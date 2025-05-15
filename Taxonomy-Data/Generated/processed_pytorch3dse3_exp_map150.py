import torch

def _so3_exp_map(omega, eps=1e-5):
    """
    Compute the exponential map from so(3) to SO(3).
    omega: (minibatch, 3) tensor representing the axis-angle rotation vectors.
    eps: small value to avoid division by zero.
    Returns: (minibatch, 3, 3) tensor representing rotation matrices.
    """
    theta = torch.norm(omega, dim=1, keepdim=True)
    theta_clamped = torch.clamp(theta, min=eps)
    omega_normalized = omega / theta_clamped

    # Skew-symmetric matrix
    K = torch.zeros((omega.size(0), 3, 3), device=omega.device)
    K[:, 0, 1] = -omega_normalized[:, 2]
    K[:, 0, 2] = omega_normalized[:, 1]
    K[:, 1, 0] = omega_normalized[:, 2]
    K[:, 1, 2] = -omega_normalized[:, 0]
    K[:, 2, 0] = -omega_normalized[:, 1]
    K[:, 2, 1] = omega_normalized[:, 0]

    I = torch.eye(3, device=omega.device).unsqueeze(0)
    R = I + torch.sin(theta_clamped).unsqueeze(-1) * K + (1 - torch.cos(theta_clamped)).unsqueeze(-1) * torch.bmm(K, K)
    return R

def _se3_V_matrix(omega, eps=1e-5):
    """
    Compute the V matrix for SE(3) exponential map.
    omega: (minibatch, 3) tensor representing the axis-angle rotation vectors.
    eps: small value to avoid division by zero.
    Returns: (minibatch, 3, 3) tensor representing the V matrices.
    """
    theta = torch.norm(omega, dim=1, keepdim=True)
    theta_clamped = torch.clamp(theta, min=eps)
    omega_normalized = omega / theta_clamped

    # Skew-symmetric matrix
    K = torch.zeros((omega.size(0), 3, 3), device=omega.device)
    K[:, 0, 1] = -omega_normalized[:, 2]
    K[:, 0, 2] = omega_normalized[:, 1]
    K[:, 1, 0] = omega_normalized[:, 2]
    K[:, 1, 2] = -omega_normalized[:, 0]
    K[:, 2, 0] = -omega_normalized[:, 1]
    K[:, 2, 1] = omega_normalized[:, 0]

    I = torch.eye(3, device=omega.device).unsqueeze(0)
    theta_clamped_sq = theta_clamped ** 2
    V = I + ((1 - torch.cos(theta_clamped)) / theta_clamped_sq).unsqueeze(-1) * K + ((theta_clamped - torch.sin(theta_clamped)) / (theta_clamped_sq * theta_clamped)).unsqueeze(-1) * torch.bmm(K, K)
    return V

def se3_exp_map(log_transform, eps=1e-5):
    """
    Convert a batch of logarithmic representations of SE(3) matrices to a batch of 4x4 SE(3) matrices.
    log_transform: (minibatch, 6) tensor representing the logarithmic representations of SE(3) matrices.
    eps: small value for clamping the rotation logarithm.
    Returns: (minibatch, 4, 4) tensor representing the SE(3) transformation matrices.
    """
    if log_transform.ndim != 2 or log_transform.size(1) != 6:
        raise ValueError("Input log_transform must have shape (minibatch, 6)")

    omega = log_transform[:, :3]
    v = log_transform[:, 3:]

    R = _so3_exp_map(omega, eps)
    V = _se3_V_matrix(omega, eps)
    t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1)

    T = torch.eye(4, device=log_transform.device).unsqueeze(0).repeat(log_transform.size(0), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return T