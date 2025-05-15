import torch
from torch import Tensor

class PinholeCamera:
    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    def scale(self, scale_factor: Tensor) -> 'PinholeCamera':
        # Ensure scale_factor is a tensor of shape (B) or (1)
        if scale_factor.dim() == 0:
            scale_factor = scale_factor.unsqueeze(0)

        # Scale the intrinsic parameters
        # Assuming intrinsics is a 3x3 matrix with fx, fy, cx, cy
        scaled_intrinsics = self._intrinsics.clone()
        scaled_intrinsics[..., 0, 0] *= scale_factor  # fx
        scaled_intrinsics[..., 1, 1] *= scale_factor  # fy
        scaled_intrinsics[..., 0, 2] *= scale_factor  # cx
        scaled_intrinsics[..., 1, 2] *= scale_factor  # cy

        # Scale the image dimensions
        scaled_height = self.height * scale_factor
        scaled_width = self.width * scale_factor

        # Return a new instance of PinholeCamera with scaled parameters
        return PinholeCamera(scaled_intrinsics, self._extrinsics, scaled_height, scaled_width)