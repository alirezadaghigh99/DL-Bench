import torch

def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched).

    For 1, 2, and 3 dimensions this uses the formulae.
    For larger matrices, it uses blockwise inversion to reduce to
    smaller matrices.
    """
    if M.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions")

    *batch_dims, n, m = M.shape
    if n != m:
        raise ValueError("The rightmost two dimensions must be square")

    if n == 1:
        # 1x1 matrix inversion
        return 1.0 / M

    elif n == 2:
        # 2x2 matrix inversion
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
        inv_det = 1.0 / det
        inv_M = torch.empty_like(M)
        inv_M[..., 0, 0] = M[..., 1, 1] * inv_det
        inv_M[..., 0, 1] = -M[..., 0, 1] * inv_det
        inv_M[..., 1, 0] = -M[..., 1, 0] * inv_det
        inv_M[..., 1, 1] = M[..., 0, 0] * inv_det
        return inv_M

    elif n == 3:
        # 3x3 matrix inversion using the adjugate method
        inv_M = torch.empty_like(M)
        det = (
            M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0])
        )
        inv_det = 1.0 / det

        inv_M[..., 0, 0] = (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1]) * inv_det
        inv_M[..., 0, 1] = (M[..., 0, 2] * M[..., 2, 1] - M[..., 0, 1] * M[..., 2, 2]) * inv_det
        inv_M[..., 0, 2] = (M[..., 0, 1] * M[..., 1, 2] - M[..., 0, 2] * M[..., 1, 1]) * inv_det
        inv_M[..., 1, 0] = (M[..., 1, 2] * M[..., 2, 0] - M[..., 1, 0] * M[..., 2, 2]) * inv_det
        inv_M[..., 1, 1] = (M[..., 0, 0] * M[..., 2, 2] - M[..., 0, 2] * M[..., 2, 0]) * inv_det
        inv_M[..., 1, 2] = (M[..., 0, 2] * M[..., 1, 0] - M[..., 0, 0] * M[..., 1, 2]) * inv_det
        inv_M[..., 2, 0] = (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]) * inv_det
        inv_M[..., 2, 1] = (M[..., 0, 1] * M[..., 2, 0] - M[..., 0, 0] * M[..., 2, 1]) * inv_det
        inv_M[..., 2, 2] = (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]) * inv_det
        return inv_M

    else:
        # For larger matrices, use torch.linalg.inv or blockwise inversion
        if sym:
            return torch.linalg.inv(M)
        else:
            return torch.linalg.inv(M)

