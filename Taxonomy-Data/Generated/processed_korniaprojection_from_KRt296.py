import torch
from torch import Tensor

def projection_from_KRt(K: Tensor, R: Tensor, t: Tensor) -> Tensor:
    # Check the input shapes
    if K.shape[-2:] != (3, 3):
        raise AssertionError(f"Expected K to have shape (B, 3, 3), but got {K.shape}")
    if R.shape[-2:] != (3, 3):
        raise AssertionError(f"Expected R to have shape (B, 3, 3), but got {R.shape}")
    if t.shape[-2:] != (3, 1):
        raise AssertionError(f"Expected t to have shape (B, 3, 1), but got {t.shape}")
    if not len(K.shape) == len(R.shape) == len(t.shape):
        raise AssertionError("K, R, and t must have the same number of dimensions")

    # Concatenate R and t to form [R|t]
    Rt = torch.cat((R, t), dim=-1)  # Shape: (B, 3, 4)

    # Compute the projection matrix P
    P = torch.bmm(K, Rt)  # Shape: (B, 3, 4)

    # Add a row [0, 0, 0, 1] to make P a 4x4 matrix
    batch_size = P.shape[0]
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=P.dtype, device=P.device).view(1, 1, 4)
    bottom_row = bottom_row.expand(batch_size, 1, 4)  # Shape: (B, 1, 4)

    # Concatenate the bottom row to P
    P = torch.cat((P, bottom_row), dim=1)  # Shape: (B, 4, 4)

    return P