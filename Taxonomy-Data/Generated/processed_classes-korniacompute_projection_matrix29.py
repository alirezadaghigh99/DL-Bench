def compute_projection_matrix(self, pinhole_src: PinholeCamera) -> DepthWarper:
    # Check if the destination and source pinhole cameras are of the correct type
    if not isinstance(self._pinhole_dst, PinholeCamera):
        raise TypeError(
            f"Member self._pinhole_dst expected to be of class PinholeCamera. Got {type(self._pinhole_dst)}"
        )
    if not isinstance(pinhole_src, PinholeCamera):
        raise TypeError(f"Argument pinhole_src expected to be of class PinholeCamera. Got {type(pinhole_src)}")

    # Set the source pinhole camera
    self._pinhole_src = pinhole_src

    # Get the intrinsic matrices for the source and destination cameras
    K_dst = self._pinhole_dst.intrinsics  # 3x3 intrinsic matrix of the destination camera
    K_src = pinhole_src.intrinsics        # 3x3 intrinsic matrix of the source camera

    # Get the extrinsic matrices (rotation and translation) for the source and destination cameras
    R_dst, t_dst = self._pinhole_dst.extrinsics  # 3x3 rotation and 3x1 translation of the destination camera
    R_src, t_src = pinhole_src.extrinsics        # 3x3 rotation and 3x1 translation of the source camera

    # Compute the relative rotation and translation from source to destination
    R_src_to_dst = R_dst @ R_src.T
    t_src_to_dst = t_dst - R_src_to_dst @ t_src

    # Construct the transformation matrix from source to destination
    T_src_to_dst = torch.eye(4, device=K_dst.device, dtype=K_dst.dtype)
    T_src_to_dst[:3, :3] = R_src_to_dst
    T_src_to_dst[:3, 3] = t_src_to_dst.squeeze()

    # Compute the projection matrix from source to destination
    P_src_to_dst = K_dst @ T_src_to_dst[:3, :] @ torch.inverse(K_src)

    # Store the projection matrix
    self._dst_proj_src = P_src_to_dst

    return self