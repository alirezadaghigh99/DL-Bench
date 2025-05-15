def projection_from_KRt(K: Tensor, R: Tensor, t: Tensor) -> Tensor:
    r"""Get the projection matrix P from K, R and t.

    This function estimate the projection matrix by solving the following equation: :math:`P = K * [R|t]`.

    Args:
       K: the camera matrix with the intrinsics with shape :math:`(B, 3, 3)`.
       R: The rotation matrix with shape :math:`(B, 3, 3)`.
       t: The translation vector with shape :math:`(B, 3, 1)`.

    Returns:
       The projection matrix P with shape :math:`(B, 4, 4)`.
    """
    KORNIA_CHECK_SHAPE(K, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(R, ["*", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["*", "3", "1"])
    if not len(K.shape) == len(R.shape) == len(t.shape):
        raise AssertionError

    Rt = concatenate([R, t], dim=-1)  # 3x4
    Rt_h = pad(Rt, [0, 0, 0, 1], "constant", 0.0)  # 4x4
    Rt_h[..., -1, -1] += 1.0

    K_h = pad(K, [0, 1, 0, 1], "constant", 0.0)  # 4x4
    K_h[..., -1, -1] += 1.0

    return K @ Rt