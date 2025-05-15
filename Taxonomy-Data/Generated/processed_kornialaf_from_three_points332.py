import torch
from torch import Tensor

def laf_from_three_points(threepts: Tensor) -> Tensor:
    """Convert three points to local affine frame.

    Order is (0,0), (0, 1), (1, 0).

    Args:
        threepts: :math:`(B, N, 2, 3)`.

    Returns:
        laf :math:`(B, N, 2, 3)`.
    """
    # Ensure the input is a float tensor
    threepts = threepts.float()

    # Extract the three points
    p0 = threepts[..., 0]  # (B, N, 2)
    p1 = threepts[..., 1]  # (B, N, 2)
    p2 = threepts[..., 2]  # (B, N, 2)

    # Compute the vectors from p0 to p1 and p0 to p2
    v1 = p1 - p0  # (B, N, 2)
    v2 = p2 - p0  # (B, N, 2)

    # Normalize v1 to get the first basis vector
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)  # (B, N, 1)
    e1 = v1 / v1_norm  # (B, N, 2)

    # Compute the second basis vector using Gram-Schmidt process
    dot_product = torch.sum(e1 * v2, dim=-1, keepdim=True)  # (B, N, 1)
    v2_orthogonal = v2 - dot_product * e1  # (B, N, 2)
    v2_orthogonal_norm = torch.norm(v2_orthogonal, dim=-1, keepdim=True)  # (B, N, 1)
    e2 = v2_orthogonal / v2_orthogonal_norm  # (B, N, 2)

    # Construct the local affine frame
    laf = torch.stack([e1, e2, p0], dim=-1)  # (B, N, 2, 3)

    return laf