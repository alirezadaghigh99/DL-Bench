    def project(self, point_3d: Tensor) -> Tensor:
        r"""Project a 3d point in world coordinates onto the 2d camera plane.

        Args:
            point3d: tensor containing the 3d points to be projected
                to the camera plane. The shape of the tensor can be :math:`(*, 3)`.

        Returns:
            tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> X = torch.rand(1, 3)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.project(X)
            tensor([[5.6088, 8.6827]])
        """
        P = self.intrinsics @ self.extrinsics
        return convert_points_from_homogeneous(transform_points(P, point_3d))