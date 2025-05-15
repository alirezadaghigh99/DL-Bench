def depth_to_normals(depth: Tensor, camera_matrix: Tensor, normalize_points: bool = False) -> Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.
        normalize_points: whether to normalize the pointcloud. This must be set to `True` when the depth is
        represented as the Euclidean ray length from the camera position.

    Return:
        tensor with a normal surface vector per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_normals(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(depth)
    KORNIA_CHECK_IS_TENSOR(camera_matrix)
    KORNIA_CHECK_SHAPE(depth, ["B", "1", "H", "W"])
    KORNIA_CHECK_SHAPE(camera_matrix, ["B", "3", "3"])

    # compute the 3d points from depth
    xyz: Tensor = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return kornia_ops.normalize(normals, dim=1, p=2)