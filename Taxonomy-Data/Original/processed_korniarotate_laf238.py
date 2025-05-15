def rotate_laf(LAF: Tensor, angles_degrees: Tensor) -> Tensor:
    """Apply additional rotation to the LAFs. Compared to `set_laf_orientation`, the resulting rotation is original
    LAF orientation plus angles_degrees.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    B, N = LAF.shape[:2]
    rotmat = angle_to_rotation_matrix(angles_degrees).view(B * N, 2, 2)
    out_laf = LAF.clone()
    out_laf[:, :, :2, :2] = torch.bmm(LAF[:, :, :2, :2].reshape(B * N, 2, 2), rotmat).reshape(B, N, 2, 2)
    return out_laf