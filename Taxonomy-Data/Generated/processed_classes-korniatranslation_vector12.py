import torch
from torch import Tensor

class PinholeCamera:
    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    def translation_vector(self) -> Tensor:
        # Assuming extrinsics is of shape (B, 4, 4)
        # Extract the translation vector from the extrinsics matrix
        translation = self._extrinsics[:, :3, 3].unsqueeze(-1)
        return translation

