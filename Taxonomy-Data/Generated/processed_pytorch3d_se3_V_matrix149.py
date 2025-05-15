import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    """
    Computes the "V" matrix used in SE(3) transformations.

    Parameters:
    - log_rotation (torch.Tensor): The logarithm of the rotation matrix.
    - log_rotation_hat (torch.Tensor): The skew-symmetric matrix derived from `log_rotation`.
    - log_rotation_hat_square (torch.Tensor): The square of the skew-symmetric matrix.
    - rotation_angles (torch.Tensor): The angles of rotation.
    - eps (float, optional): A small value for numerical stability, defaulting to 1e-4.

    Returns:
    - V (torch.Tensor): The computed "V" matrix.
    """
    # Ensure rotation_angles is at least 1D
    rotation_angles = rotation_angles.unsqueeze(-1) if rotation_angles.dim() == 0 else rotation_angles

    # Compute terms for the V matrix
    angle_squared = rotation_angles ** 2
    sin_angle = torch.sin(rotation_angles)
    cos_angle = torch.cos(rotation_angles)

    # Compute coefficients
    A = sin_angle / (rotation_angles + eps)
    B = (1 - cos_angle) / (angle_squared + eps)
    C = (1 - A) / (angle_squared + eps)

    # Compute the V matrix
    V = torch.eye(3, device=log_rotation.device) + B * log_rotation_hat + C * log_rotation_hat_square

    return V

