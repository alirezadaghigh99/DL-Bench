def distort_points_kannala_brandt(projected_points_in_camera_z1_plane: Tensor, params: Tensor) -> Tensor:
    r"""Distort one or more points from the canonical z=1 plane into the camera frame using the Kannala-Brandt
    model.

    Args:
        projected_points_in_camera_z1_plane: Tensor representing the points to distort with shape (..., 2).
        params: Tensor representing the parameters of the Kannala-Brandt distortion model with shape (..., 8).

    Returns:
        Tensor representing the distorted points with shape (..., 2).

    Example:
        >>> points = torch.tensor([319.5, 239.5])  # center of a 640x480 image
        >>> params = torch.tensor([1000.0, 1000.0, 320.0, 280.0, 0.1, 0.01, 0.001, 0.0001])
        >>> distort_points_kannala_brandt(points, params)
        tensor([1982.6832, 1526.3619])
    """
    KORNIA_CHECK_SHAPE(projected_points_in_camera_z1_plane, ["*", "2"])
    KORNIA_CHECK_SHAPE(params, ["*", "8"])

    x = projected_points_in_camera_z1_plane[..., 0]
    y = projected_points_in_camera_z1_plane[..., 1]

    radius_sq = x**2 + y**2

    # TODO: we can optimize this by passing the radius_sq to the impl functions. Check if it's worth it.
    distorted_points = ops.where(
        radius_sq[..., None] > 1e-8,
        _distort_points_kannala_brandt_impl(
            projected_points_in_camera_z1_plane,
            params,
        ),
        distort_points_affine(projected_points_in_camera_z1_plane, params[..., :4]),
    )

    return distorted_points