import torch

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculate the rotation angles from a batch of 3x3 rotation matrices.

    Args:
        R (torch.Tensor): A batch of 3x3 rotation matrices of shape (N, 3, 3).
        eps (float): A small epsilon value to handle numerical stability.
        cos_angle (bool): If True, return the cosine of the angle instead of the angle itself.
        cos_bound (float): A small value to clamp the cosine of the angle to avoid numerical issues.

    Returns:
        torch.Tensor: A tensor of rotation angles (or their cosines) of shape (N,).
    """
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("Input must be a batch of 3x3 matrices with shape (N, 3, 3).")

    # Calculate the trace of each matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Ensure the trace is within the valid range [-1, 3]
    if not torch.all((trace >= -1.0 - eps) & (trace <= 3.0 + eps)):
        raise ValueError("Trace of each matrix must be in the range [-1, 3].")

    # Calculate the cosine of the rotation angle
    cos_theta = (trace - 1) / 2

    # Clamp the cosine to the range [-1 + cos_bound, 1 - cos_bound]
    cos_theta = torch.clamp(cos_theta, -1.0 + cos_bound, 1.0 - cos_bound)

    if cos_angle:
        return cos_theta

    # Calculate the rotation angle in radians
    angle = torch.acos(cos_theta)

    return angle