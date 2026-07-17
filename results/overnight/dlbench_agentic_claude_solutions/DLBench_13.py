import torch

def _safe_det_3x3(t: torch.Tensor):
    """Create a Python function called _safe_det_3x3 that calculates the determinant of a batch of 3x3 matrices. The input is a torch.Tensor t of shape (N, 3, 3) where N is the number of matrices in the batch. The function returns a torch.Tensor of shape (N) containing the determinants of the input matrices. Note that the result of this function may differ slightly from the torch.det() function, with potential differences in the last significant digit."""
    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] - t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] - t[..., 2, 0] * t[..., 1, 1])
    )

    return det
