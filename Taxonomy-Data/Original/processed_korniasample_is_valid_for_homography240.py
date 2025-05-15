def sample_is_valid_for_homography(points1: Tensor, points2: Tensor) -> Tensor:
    """Function, which implements oriented constraint check from :cite:`Marquez-Neila2015`.

    Analogous to https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/usac/degeneracy.cpp#L88

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, 4, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, 4, 2)`.

    Returns:
        Mask with the minimal sample is good for homography estimation:math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    KORNIA_CHECK_SHAPE(points1, ["B", "4", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "4", "2"])
    device = points1.device
    idx_perm = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long, device=device)
    points_src_h = convert_points_to_homogeneous(points1)
    points_dst_h = convert_points_to_homogeneous(points2)

    src_perm = points_src_h[:, idx_perm]
    dst_perm = points_dst_h[:, idx_perm]
    left_sign = (
        torch.cross(src_perm[..., 1:2, :], src_perm[..., 2:3, :]) @ src_perm[..., 0:1, :].permute(0, 1, 3, 2)
    ).sign()
    right_sign = (
        torch.cross(dst_perm[..., 1:2, :], dst_perm[..., 2:3, :]) @ dst_perm[..., 0:1, :].permute(0, 1, 3, 2)
    ).sign()
    sample_is_valid = (left_sign == right_sign).view(-1, 4).min(dim=1)[0]
    return sample_is_valid