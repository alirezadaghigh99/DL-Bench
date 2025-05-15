    def translation_vector(self) -> Tensor:
        r"""Return the translation vector from the extrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 1)`.
        """
        return self.extrinsics[..., :3, -1:]