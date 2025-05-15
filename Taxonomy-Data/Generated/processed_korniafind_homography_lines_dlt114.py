import torch

def find_homography_lines_dlt(ls1, ls2, weights=None):
    """
    Computes the homography matrix using the DLT formulation for line correspondences.

    Parameters:
    - ls1: Tensor of shape (B, N, 2, 2) representing the first set of line segments.
    - ls2: Tensor of shape (B, N, 2, 2) representing the second set of line segments.
    - weights: Optional tensor of shape (B, N) representing weights for each line correspondence.

    Returns:
    - homographies: Tensor of shape (B, 3, 3) representing the computed homography matrices.
    """
    B, N, _, _ = ls1.shape

    if weights is None:
        weights = torch.ones((B, N), dtype=ls1.dtype, device=ls1.device)

    # Convert line segments to homogeneous line representations
    def line_to_homogeneous(l):
        p1, p2 = l[:, :, 0, :], l[:, :, 1, :]
        return torch.cross(
            torch.cat([p1, torch.ones((B, N, 1), dtype=l.dtype, device=l.device)], dim=-1),
            torch.cat([p2, torch.ones((B, N, 1), dtype=l.dtype, device=l.device)], dim=-1),
            dim=-1
        )

    L1 = line_to_homogeneous(ls1)
    L2 = line_to_homogeneous(ls2)

    # Construct the matrix A for each batch
    A = torch.zeros((B, 2 * N, 9), dtype=ls1.dtype, device=ls1.device)

    for i in range(N):
        l1 = L1[:, i, :]
        l2 = L2[:, i, :]
        w = weights[:, i].unsqueeze(-1)

        A[:, 2 * i, :] = w * torch.cat([l1[:, 0:1] * l2, l1[:, 1:2] * l2, l1[:, 2:3] * l2], dim=-1)
        A[:, 2 * i + 1, :] = w * torch.cat([l1[:, 0:1] * l2, l1[:, 1:2] * l2, l1[:, 2:3] * l2], dim=-1)

    # Solve the system using SVD
    _, _, V = torch.svd(A)
    H = V[:, -1, :].reshape(B, 3, 3)

    return H

