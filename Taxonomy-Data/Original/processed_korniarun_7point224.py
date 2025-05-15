def run_7point(points1: Tensor, points2: Tensor) -> Tensor:
    r"""Compute the fundamental matrix using the 7-point algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3), Valid values of m are 1, 2 or 3`
    """
    KORNIA_CHECK_SHAPE(points1, ["B", "7", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "7", "2"])

    batch_size = points1.shape[0]

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = ones_like(x1)
    # form a linear system: which represents
    # the equation (x2[i], 1)*F*(x1[i], 1) = 0
    X = concatenate([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], -1)  # BxNx9

    # X * Fmat = 0 is singular (7 equations for 9 variables)
    # solving for nullspace of X to get two F
    ####### unstable failing gradcheck
    # _, _, v = torch.linalg.svd(X)
    _, _, v = _torch_svd_cast(X)

    # last two singular vector as a basic of the space
    f1 = v[..., 7].view(-1, 3, 3)
    f2 = v[..., 8].view(-1, 3, 3)

    # lambda*f1 + mu*f2 is an arbitrary fundamental matrix
    # f ~ lambda*f1 + (1 - lambda)*f2
    # det(f) = det(lambda*f1 + (1-lambda)*f2), find lambda
    # form a cubic equation
    # finding the coefficients of cubic polynomial (coeffs)

    coeffs = zeros((batch_size, 4), device=v.device, dtype=v.dtype)

    f1_det = torch.linalg.det(f1)
    f2_det = torch.linalg.det(f2)
    coeffs[:, 0] = f1_det
    coeffs[:, 1] = torch.einsum("bii->b", f2 @ safe_inverse_with_mask(f1)[0]) * f1_det
    coeffs[:, 2] = torch.einsum("bii->b", f1 @ safe_inverse_with_mask(f2)[0]) * f2_det
    coeffs[:, 3] = f2_det

    # solve the cubic equation, there can be 1 to 3 roots
    # roots = torch.tensor(np.roots(coeffs.numpy()))
    roots = solve_cubic(coeffs)

    fmatrix = zeros((batch_size, 3, 3, 3), device=v.device, dtype=v.dtype)
    valid_root_mask = (torch.count_nonzero(roots, dim=1) < 3) | (torch.count_nonzero(roots, dim=1) > 1)

    _lambda = roots
    _mu = torch.ones_like(_lambda)

    _s = f1[valid_root_mask, 2, 2].unsqueeze(dim=1) * roots[valid_root_mask] + f2[valid_root_mask, 2, 2].unsqueeze(
        dim=1
    )
    # _s_non_zero_mask = torch.abs(_s ) > 1e-16
    _s_non_zero_mask = ~torch.isclose(_s, torch.tensor(0.0, device=v.device, dtype=v.dtype))

    _mu[_s_non_zero_mask] = 1.0 / _s[_s_non_zero_mask]
    _lambda[_s_non_zero_mask] = _lambda[_s_non_zero_mask] * _mu[_s_non_zero_mask]

    f1_expanded = f1.unsqueeze(1).expand(batch_size, 3, 3, 3)
    f2_expanded = f2.unsqueeze(1).expand(batch_size, 3, 3, 3)

    fmatrix[valid_root_mask] = (
        f1_expanded[valid_root_mask] * _lambda[valid_root_mask, :, None, None]
        + f2_expanded[valid_root_mask] * _mu[valid_root_mask, :, None, None]
    )

    mat_ind = zeros(3, 3, dtype=torch.bool)
    mat_ind[2, 2] = True
    fmatrix[_s_non_zero_mask, mat_ind] = 1.0
    fmatrix[~_s_non_zero_mask, mat_ind] = 0.0

    trans1_exp = transform1[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)
    trans2_exp = transform2[valid_root_mask].unsqueeze(1).expand(-1, fmatrix.shape[2], -1, -1)

    fmatrix[valid_root_mask] = torch.matmul(
        trans2_exp.transpose(-2, -1), torch.matmul(fmatrix[valid_root_mask], trans1_exp)
    )

    return normalize_transformation(fmatrix)