def project_points_z1(points_in_camera: Tensor) -> Tensor:
    r"""Project one or more points from the camera frame into the canonical z=1 plane through perspective division.

    .. math::

        \begin{bmatrix} u \\ v \\ w \end{bmatrix} =
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} / z

    .. note::

        This function has a precondition that the points are in front of the camera, i.e. z > 0.
        If this is not the case, the points will be projected to the canonical plane, but the resulting
        points will be behind the camera and causing numerical issues for z == 0.

    Args:
        points_in_camera: Tensor representing the points to project with shape (..., 3).

    Returns:
        Tensor representing the projected points with shape (..., 2).

    Example:
        >>> points = torch.tensor([1., 2., 3.])
        >>> project_points_z1(points)
        tensor([0.3333, 0.6667])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "3"])
    return points_in_camera[..., :2] / points_in_camera[..., 2:3]