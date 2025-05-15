def set_laf_orientation(LAF: Tensor, angles_degrees: Tensor) -> Tensor:
    """Change the orientation of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`
        angles: :math:`(B, N, 1)` in degrees.

    Returns:
        LAF oriented with angles :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    B, N = LAF.shape[:2]
    ori = get_laf_orientation(LAF).reshape_as(angles_degrees)
    return rotate_laf(LAF, angles_degrees - ori)