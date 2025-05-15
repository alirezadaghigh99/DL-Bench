def rinverse(M, sym=False):
    """Matrix inversion of rightmost dimensions (batched).

    For 1, 2, and 3 dimensions this uses the formulae.
    For larger matrices, it uses blockwise inversion to reduce to
    smaller matrices.
    """
    assert M.shape[-1] == M.shape[-2]
    if M.shape[-1] == 1:
        return 1.0 / M
    elif M.shape[-1] == 2:
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 1, 0] * M[..., 0, 1]
        inv = torch.empty_like(M)
        inv[..., 0, 0] = M[..., 1, 1]
        inv[..., 1, 1] = M[..., 0, 0]
        inv[..., 0, 1] = -M[..., 0, 1]
        inv[..., 1, 0] = -M[..., 1, 0]
        return inv / det.unsqueeze(-1).unsqueeze(-1)
    elif M.shape[-1] == 3:
        return inv3d(M, sym=sym)
    else:
        return torch.inverse(M)