def depth_from_point(R: Tensor, t: Tensor, X: Tensor) -> Tensor:
    r"""Return the depth of a point transformed by a rigid transform.

    Args:
       R: The rotation matrix with shape :math:`(*, 3, 3)`.
       t: The translation vector with shape :math:`(*, 3, 1)`.
       X: The 3d points with shape :math:`(*, 3)`.

    Returns:
       The depth value per point with shape :math:`(*, 1)`.
    """
    X_tmp = R @ X.transpose(-2, -1)
    X_out = X_tmp[..., 2, :] + t[..., 2, :]
    return X_out