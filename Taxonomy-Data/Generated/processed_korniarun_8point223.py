import torch

def normalize_points(points):
    """Normalize points for numerical stability."""
    mean = points.mean(dim=1, keepdim=True)
    std = points.std(dim=1, keepdim=True)
    normalized_points = (points - mean) / std
    T = torch.tensor([[1/std[0,0,0], 0, -mean[0,0,0]/std[0,0,0]],
                      [0, 1/std[0,0,1], -mean[0,0,1]/std[0,0,1]],
                      [0, 0, 1]], device=points.device)
    return normalized_points, T

def run_8point(points1, points2, weights):
    """
    Compute the fundamental matrix using the 8-point algorithm with weighted least squares.

    Args:
        points1: A set of points in the first image with a tensor shape (B, N, 2), N>=8.
        points2: A set of points in the second image with a tensor shape (B, N, 2), N>=8.
        weights: Tensor containing the weights per point correspondence with a shape of (B, N).

    Returns:
        The computed fundamental matrix with shape (B, 3, 3).
    """
    B, N, _ = points1.shape

    # Normalize points
    points1_normalized, T1 = normalize_points(points1)
    points2_normalized, T2 = normalize_points(points2)

    # Construct the design matrix A
    x1, y1 = points1_normalized[..., 0], points1_normalized[..., 1]
    x2, y2 = points2_normalized[..., 0], points2_normalized[..., 1]
    A = torch.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)], dim=-1)

    # Apply weights
    W = weights.unsqueeze(-1)
    A_weighted = W * A

    # Solve the weighted least squares problem using SVD
    _, _, V = torch.svd(A_weighted)
    F = V[..., -1].view(B, 3, 3)

    # Enforce the rank-2 constraint
    U, S, Vt = torch.svd(F)
    S[..., -1] = 0
    F_rank2 = U @ torch.diag_embed(S) @ Vt

    # Denormalize the fundamental matrix
    F_final = T2.transpose(-1, -2) @ F_rank2 @ T1

    return F_final

