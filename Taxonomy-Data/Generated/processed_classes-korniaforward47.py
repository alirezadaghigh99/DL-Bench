import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Affine(nn.Module):
    def __init__(
        self,
        angle: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
        shear: Optional[torch.Tensor] = None,
        center: Optional[torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super(Affine, self).__init__()
        self.angle = angle
        self.translation = translation
        self.scale_factor = scale_factor
        self.shear = shear
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input.size()
        
        # Default values for transformations
        angle = self.angle if self.angle is not None else torch.zeros(batch_size)
        translation = self.translation if self.translation is not None else torch.zeros(batch_size, 2)
        scale_factor = self.scale_factor if self.scale_factor is not None else torch.ones(batch_size, 2)
        shear = self.shear if self.shear is not None else torch.zeros(batch_size, 2)
        center = self.center if self.center is not None else torch.tensor([width / 2, height / 2]).repeat(batch_size, 1)

        # Convert angle from degrees to radians
        angle = angle * torch.pi / 180.0

        # Create affine transformation matrices
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Rotation matrix
        rotation_matrix = torch.stack([
            cos_a, -sin_a,
            sin_a, cos_a
        ], dim=-1).view(batch_size, 2, 2)

        # Scale matrix
        scale_matrix = torch.diag_embed(scale_factor)

        # Shear matrix
        shear_matrix = torch.stack([
            torch.ones_like(shear[:, 0]), shear[:, 0],
            shear[:, 1], torch.ones_like(shear[:, 1])
        ], dim=-1).view(batch_size, 2, 2)

        # Combine transformations: R * S * Sh
        transform_matrix = rotation_matrix @ scale_matrix @ shear_matrix

        # Adjust for center
        center_matrix = torch.eye(3).repeat(batch_size, 1, 1)
        center_matrix[:, 0, 2] = center[:, 0]
        center_matrix[:, 1, 2] = center[:, 1]

        # Translation matrix
        translation_matrix = torch.eye(3).repeat(batch_size, 1, 1)
        translation_matrix[:, 0, 2] = translation[:, 0]
        translation_matrix[:, 1, 2] = translation[:, 1]

        # Combine all transformations
        full_transform = torch.eye(3).repeat(batch_size, 1, 1)
        full_transform[:, :2, :2] = transform_matrix
        full_transform = translation_matrix @ center_matrix @ full_transform @ torch.inverse(center_matrix)

        # Extract 2x3 affine part
        affine_matrices = full_transform[:, :2, :]

        # Create grid and apply transformation
        grid = F.affine_grid(affine_matrices, input.size(), align_corners=self.align_corners)
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

        return output