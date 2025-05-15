import torch

def KORNIA_CHECK_LAF(laf, raises=False):
    """
    Check if the input tensor `laf` has the shape (B, N, 2, 3).

    Parameters:
    laf (torch.Tensor): The input tensor to check.
    raises (bool): If True, raise an exception if the shape is invalid.

    Returns:
    bool: True if the shape is valid, False otherwise.
    """
    if laf.ndim != 4 or laf.shape[2] != 2 or laf.shape[3] != 3:
        if raises:
            raise ValueError(f"Invalid LAF shape: {laf.shape}. Expected shape is (B, N, 2, 3).")
        return False
    return True

