import torch
from torch import Tensor

class PinholeCamera:
    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    def project(self, point_3d: Tensor) -> Tensor:
        # Ensure point_3d is in homogeneous coordinates
        if point_3d.shape[-1] != 4:
            ones = torch.ones((*point_3d.shape[:-1], 1), device=point_3d.device, dtype=point_3d.dtype)
            point_3d_homogeneous = torch.cat((point_3d, ones), dim=-1)
        else:
            point_3d_homogeneous = point_3d

        # Transform the 3D points from world coordinates to camera coordinates
        point_camera = torch.matmul(point_3d_homogeneous, self._extrinsics.T)

        # Project the 3D camera coordinates onto the 2D image plane
        point_image_homogeneous = torch.matmul(point_camera, self._intrinsics.T)

        # Convert from homogeneous to 2D coordinates
        u = point_image_homogeneous[..., 0] / point_image_homogeneous[..., 2]
        v = point_image_homogeneous[..., 1] / point_image_homogeneous[..., 2]

        # Stack the u, v coordinates
        uv_coordinates = torch.stack((u, v), dim=-1)

        return uv_coordinates