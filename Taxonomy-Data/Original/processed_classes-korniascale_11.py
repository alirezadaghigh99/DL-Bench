    def scale(self, scale_factor: Tensor) -> "PinholeCamera":
        r"""Scale the pinhole model.

        Args:
            scale_factor: a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            the camera model with scaled parameters.
        """
        # scale the intrinsic parameters
        intrinsics: Tensor = self.intrinsics.clone()
        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        height: Tensor = scale_factor * self.height.clone()
        width: Tensor = scale_factor * self.width.clone()
        return PinholeCamera(intrinsics, self.extrinsics, height, width)