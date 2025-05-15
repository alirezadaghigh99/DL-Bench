import torch
from torch import Tensor

class PinholeCamera:
    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    def unproject(self, point_2d: Tensor, depth: Tensor) -> Tensor:
        # Ensure the input tensors are of the correct shape
        assert point_2d.shape[-1] == 2, "point_2d should have shape (*, 2)"
        assert depth.shape[-1] == 1, "depth should have shape (*, 1)"
        
        # Get the inverse of the intrinsic matrix
        intrinsics_inv = torch.inverse(self._intrinsics)
        
        # Convert 2D points to homogeneous coordinates
        ones = torch.ones_like(point_2d[..., :1])
        pixel_homogeneous = torch.cat((point_2d, ones), dim=-1)  # Shape: (*, 3)
        
        # Compute camera coordinates
        camera_coords = (intrinsics_inv @ pixel_homogeneous.unsqueeze(-1)).squeeze(-1)  # Shape: (*, 3)
        
        # Scale by depth
        camera_coords *= depth
        
        # Convert camera coordinates to homogeneous coordinates
        camera_homogeneous = torch.cat((camera_coords, ones), dim=-1)  # Shape: (*, 4)
        
        # Transform to world coordinates using the extrinsic matrix
        world_coords_homogeneous = (self._extrinsics @ camera_homogeneous.unsqueeze(-1)).squeeze(-1)  # Shape: (*, 4)
        
        # Convert from homogeneous coordinates to 3D world coordinates
        world_coords = world_coords_homogeneous[..., :3] / world_coords_homogeneous[..., 3:4]
        
        return world_coords