def find_homography_lines_dlt(ls1: Tensor, ls2: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Compute the homography matrix using the DLT formulation for line correspondences.

    See :cite:`homolines2001` for details.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Line correspondences algorithm.

    Args:
        ls1: A set of line segments in the first image with a tensor shape :math:`(B, N, 2, 2)`.
        ls2: A set of line segments in the second image with a tensor shape :math:`(B, N, 2, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if len(ls1.shape) == 3:
        ls1 = ls1[None]
    if len(ls2.shape) == 3:
        ls2 = ls2[None]
    KORNIA_CHECK_SHAPE(ls1, ["B", "N", "2", "2"])
    KORNIA_CHECK_SHAPE(ls2, ["B", "N", "2", "2"])
    BS, N = ls1.shape[:2]
    device, dtype = _extract_device_dtype([ls1, ls2])

    points1 = ls1.reshape(BS, 2 * N, 2)
    points2 = ls2.reshape(BS, 2 * N, 2)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)
    lst1, le1 = torch.chunk(points1_norm, dim=1, chunks=2)
    lst2, le2 = torch.chunk(points2_norm, dim=1, chunks=2)

    xs1, ys1 = torch.chunk(lst1, dim=-1, chunks=2)  # BxNx1
    xs2, ys2 = torch.chunk(lst2, dim=-1, chunks=2)  # BxNx1
    xe1, ye1 = torch.chunk(le1, dim=-1, chunks=2)  # BxNx1
    xe2, ye2 = torch.chunk(le2, dim=-1, chunks=2)  # BxNx1

    A = ys2 - ye2
    B = xe2 - xs2
    C = xs2 * ye2 - xe2 * ys2

    eps: float = 1e-8

    # http://diis.unizar.es/biblioteca/00/09/000902.pdf
    ax = torch.cat([A * xs1, A * ys1, A, B * xs1, B * ys1, B, C * xs1, C * ys1, C], dim=-1)
    ay = torch.cat([A * xe1, A * ye1, A, B * xe1, B * ye1, B, C * xe1, C * ye1, C], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not ((len(weights.shape) == 2) and (weights.shape == ls1.shape[:2])):
            raise AssertionError(weights.shape)
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    try:
        _, _, V = _torch_svd_cast(A)
    except RuntimeError:
        warnings.warn("SVD did not converge", RuntimeWarning)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)
    H = safe_inverse_with_mask(transform2)[0] @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm