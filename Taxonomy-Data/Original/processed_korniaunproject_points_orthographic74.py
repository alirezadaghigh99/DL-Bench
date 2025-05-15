def unproject_points_orthographic(points_in_camera: Tensor, extension: Tensor) -> Tensor:
    r"""Unproject one or more points from the canonical z=1 plane into the camera frame.

    .. math::
        \begin{bmatrix} x \\ y \\ z \end{bmatrix} =
        \begin{bmatrix} u \\ v \\ w \end{bmatrix}

    Args:
        points_in_camera: Tensor representing the points to unproject with shape (..., 2).
        extension: Tensor representing the extension of the points to unproject with shape (..., 1).

    Returns:
        Tensor representing the unprojected points with shape (..., 3).

    Example:
        >>> points = torch.tensor([1., 2.])
        >>> extension = torch.tensor([3.])
        >>> unproject_points_orthographic(points, extension)
        tensor([1., 2., 3.])
    """
    KORNIA_CHECK_SHAPE(points_in_camera, ["*", "2"])

    if len(points_in_camera.shape) != len(extension.shape):
        extension = extension[..., None]

    return ops.concatenate([points_in_camera, extension], dim=-1)