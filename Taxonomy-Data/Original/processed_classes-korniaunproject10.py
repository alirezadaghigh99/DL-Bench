    def unproject(self, point_2d: Tensor, depth: Tensor) -> Tensor:
        r"""Unproject a 2d point in 3d.

        Transform coordinates in the pixel frame to the world frame.

        Args:
            point2d: tensor containing the 2d to be projected to
                world coordinates. The shape of the tensor can be :math:`(*, 2)`.
            depth: tensor containing the depth value of each 2d
                points. The tensor shape must be equal to point2d :math:`(*, 1)`.
            normalize: whether to normalize the pointcloud. This
                must be set to `True` when the depth is represented as the Euclidean
                ray length from the camera position.

        Returns:
            tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> x = torch.rand(1, 2)
            >>> depth = torch.ones(1, 1)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.unproject(x, depth)
            tensor([[0.4963, 0.7682, 1.0000]])
        """
        P = self.intrinsics @ self.extrinsics
        P_inv = _torch_inverse_cast(P)
        return transform_points(P_inv, convert_points_to_homogeneous(point_2d) * depth)