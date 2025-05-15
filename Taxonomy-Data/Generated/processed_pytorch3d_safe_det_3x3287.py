import torch

def _safe_det_3x3(t: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor has the correct shape
    assert t.shape[1:] == (3, 3), "Each matrix must be 3x3 in size."
    
    # Extract individual elements of the 3x3 matrices
    a = t[:, 0, 0]
    b = t[:, 0, 1]
    c = t[:, 0, 2]
    d = t[:, 1, 0]
    e = t[:, 1, 1]
    f = t[:, 1, 2]
    g = t[:, 2, 0]
    h = t[:, 2, 1]
    i = t[:, 2, 2]
    
    # Calculate the determinant using the formula
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    
    return det

