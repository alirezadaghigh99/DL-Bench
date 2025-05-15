import torch
import torch.nn as nn
import torch.nn.functional as F

class Translate(nn.Module):
    def __init__(self, translation: torch.Tensor, mode='bilinear', padding_mode='zeros', align_corners=False):
        super(Translate, self).__init__()
        self.translation = translation
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get the translation values
        tx, ty = self.translation

        # Normalize the translation values to the range [-1, 1]
        # Assuming input is of shape (N, C, H, W)
        N, C, H, W = input.size()
        tx_normalized = 2.0 * tx / W
        ty_normalized = 2.0 * ty / H

        # Create the affine transformation matrix
        theta = torch.tensor([
            [1, 0, tx_normalized],
            [0, 1, ty_normalized]
        ], dtype=input.dtype, device=input.device).unsqueeze(0).repeat(N, 1, 1)

        # Generate the grid
        grid = F.affine_grid(theta, input.size(), align_corners=self.align_corners)

        # Sample the input tensor using the grid
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

        return output