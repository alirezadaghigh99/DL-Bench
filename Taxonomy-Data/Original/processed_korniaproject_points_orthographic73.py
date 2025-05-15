def project_points_orthographic(points_in_camera: Tensor) -> Tensor:
    r"""Project one or more points from the camera frame into the canonical z=1 plane through orthographic
    projection.

    .. math::
        \begin{bmatrix} u \\ v \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix}


    Args:
        points_in_camera: Tensor representing the points to project.

    Returns:
        Tensor representing the projected points.

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_orthographic(points)
        tensor([1., 2.])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2]