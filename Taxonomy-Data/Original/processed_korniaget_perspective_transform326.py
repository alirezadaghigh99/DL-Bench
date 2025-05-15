def get_perspective_transform(points_src: Tensor, points_dst: Tensor) -> Tensor:
    r"""Calculate a perspective transform from four pairs of the corresponding points.

    The algorithm is a vanilla implementation of the Direct Linear transform (DLT).
    See more: https://www.cs.cmu.edu/~16385/s17/Slides/10.2_2D_Alignment__DLT.pdf

    The function calculates the matrix of a perspective transform that maps from
    the source to destination points:

    .. math::

        \begin{bmatrix}
        x^{'} \\
        y^{'} \\
        1 \\
        \end{bmatrix}
        =
        \begin{bmatrix}
        h_1 & h_2 & h_3 \\
        h_4 & h_5 & h_6 \\
        h_7 & h_8 & h_9 \\
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
        x \\
        y \\
        1 \\
        \end{bmatrix}

    Args:
        points_src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 4, 2)`.
        points_dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape :math:`(B, 4, 2)`.

    Returns:
        the perspective transformation with shape :math:`(B, 3, 3)`.

    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.

    Example:
        >>> x1 = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
        >>> x2 = torch.tensor([[[1., 0.], [0., 0.], [0., 1.], [1., 1.]]])
        >>> x2_trans_x1 = get_perspective_transform(x1, x2)
    """
    KORNIA_CHECK_SHAPE(points_src, ["B", "4", "2"])
    KORNIA_CHECK_SHAPE(points_dst, ["B", "4", "2"])
    KORNIA_CHECK(points_src.shape == points_dst.shape, "Source data shape must match Destination data shape.")
    KORNIA_CHECK(points_src.dtype == points_dst.dtype, "Source data type must match Destination data type.")

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence

    # create the lhs tensor with shape # Bx8x8
    B: int = points_src.shape[0]  # batch_size

    A = torch.empty(B, 8, 8, device=points_src.device, dtype=points_src.dtype)

    # we need to perform in batch
    _zeros = zeros(B, device=points_src.device, dtype=points_src.dtype)
    _ones = ones(B, device=points_src.device, dtype=points_src.dtype)

    for i in range(4):
        x1, y1 = points_src[..., i, 0], points_src[..., i, 1]  # Bx4
        x2, y2 = points_dst[..., i, 0], points_dst[..., i, 1]  # Bx4

        A[:, 2 * i] = stack([x1, y1, _ones, _zeros, _zeros, _zeros, -x1 * x2, -y1 * x2], -1)
        A[:, 2 * i + 1] = stack([_zeros, _zeros, _zeros, x1, y1, _ones, -x1 * y2, -y1 * y2], -1)

    # the rhs tensor
    b = points_dst.view(-1, 8, 1)

    # solve the system Ax = b
    X: Tensor = _torch_solve_cast(A, b)

    # create variable to return the Bx3x3 transform
    M = torch.empty(B, 9, device=points_src.device, dtype=points_src.dtype)
    M[..., :8] = X[..., 0]  # Bx8
    M[..., -1].fill_(1)

    return M.view(-1, 3, 3)  # Bx3x3