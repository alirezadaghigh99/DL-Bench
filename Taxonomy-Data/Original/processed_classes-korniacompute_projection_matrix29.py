    def compute_projection_matrix(self, pinhole_src: PinholeCamera) -> DepthWarper:
        r"""Compute the projection matrix from the source to destination frame."""
        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError(
                f"Member self._pinhole_dst expected to be of class PinholeCamera. Got {type(self._pinhole_dst)}"
            )
        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError(f"Argument pinhole_src expected to be of class PinholeCamera. Got {type(pinhole_src)}")
        # compute the relative pose between the non reference and the reference
        # camera frames.
        dst_trans_src: Tensor = compose_transformations(
            self._pinhole_dst.extrinsics, inverse_transformation(pinhole_src.extrinsics)
        )

        # compute the projection matrix between the non reference cameras and
        # the reference.
        dst_proj_src: Tensor = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src
        return self