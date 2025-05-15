import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Rotate(nn.Module):
    def __init__(self, angle, center=None, interpolation='bilinear', padding_mode='zeros', align_corners=False):
        super(Rotate, self).__init__()
        self.angle = angle
        self.center = center
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get the dimensions of the input tensor
        n, c, h, w = input.size()
        
        # Calculate the center of rotation
        if self.center is None:
            center_x, center_y = w / 2, h / 2
        else:
            center_x, center_y = self.center
        
        # Convert angle from degrees to radians
        angle_rad = -self.angle * math.pi / 180  # Negative for anti-clockwise rotation
        
        # Calculate the rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Affine transformation matrix
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, (1 - cos_a) * center_x + sin_a * center_y],
            [sin_a, cos_a, (1 - cos_a) * center_y - sin_a * center_x]
        ], dtype=torch.float, device=input.device)
        
        # Add batch dimension to the rotation matrix
        rotation_matrix = rotation_matrix.unsqueeze(0).repeat(n, 1, 1)
        
        # Create a grid for sampling
        grid = F.affine_grid(rotation_matrix, input.size(), align_corners=self.align_corners)
        
        # Apply the grid sample
        output = F.grid_sample(input, grid, mode=self.interpolation, padding_mode=self.padding_mode, align_corners=self.align_corners)
        
        return output