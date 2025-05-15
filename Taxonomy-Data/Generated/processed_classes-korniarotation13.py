import torch
from torch import Tensor

class PinholeCamera:
    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    def rt_matrix(self) -> Tensor:
        """
        Returns the 3x4 rotation-translation matrix.

        Returns:
            Tensor of shape (B, 3, 4).
        """
        # Assuming extrinsics is of shape (B, 4, 4)
        # Extract the rotation (3x3) and translation (3x1) components
        rotation = self._extrinsics[:, :3, :3]  # Shape: (B, 3, 3)
        translation = self._extrinsics[:, :3, 3:4]  # Shape: (B, 3, 1)

        # Concatenate rotation and translation to form the 3x4 matrix
        rt_matrix = torch.cat((rotation, translation), dim=2)  # Shape: (B, 3, 4)

        return rt_matrix