import torch
import torch.nn.functional as F
from torch import nn

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    """
    Perform deformable convolution v2.

    Parameters:
    - input: Input tensor of shape (N, C_in, H_in, W_in)
    - offset: Offset tensor of shape (N, 2*kernel_h*kernel_w, H_out, W_out)
    - weight: Weight tensor of shape (C_out, C_in, kernel_h, kernel_w)
    - bias: Optional bias tensor of shape (C_out,)
    - stride: Stride of the convolution
    - padding: Zero-padding added to both sides of the input
    - dilation: Spacing between kernel elements
    - mask: Optional mask tensor of shape (N, kernel_h*kernel_w, H_out, W_out)

    Returns:
    - Output tensor of the deformable convolution
    """
    N, C_in, H_in, W_in = input.shape
    C_out, _, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    H_out = (H_in + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Ensure offset and mask dimensions are correct
    assert offset.shape == (N, 2 * kernel_h * kernel_w, H_out, W_out)
    if mask is not None:
        assert mask.shape == (N, kernel_h * kernel_w, H_out, W_out)

    # Create a grid for sampling
    grid_y, grid_x = torch.meshgrid(torch.arange(H_out), torch.arange(W_out))
    grid = torch.stack((grid_x, grid_y), 2).float().to(input.device)  # Shape: (H_out, W_out, 2)

    # Add offset to the grid
    grid = grid.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H_out, W_out, 2)
    grid = grid.repeat(N, kernel_h * kernel_w, 1, 1, 1)  # Shape: (N, kernel_h*kernel_w, H_out, W_out, 2)
    offset = offset.view(N, kernel_h * kernel_w, 2, H_out, W_out).permute(0, 1, 3, 4, 2)
    grid = grid + offset

    # Normalize grid to [-1, 1] for F.grid_sample
    grid[..., 0] = 2.0 * grid[..., 0] / max(W_out - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(H_out - 1, 1) - 1.0

    # Sample input using the grid
    input_padded = F.pad(input, (padding, padding, padding, padding))
    sampled = F.grid_sample(input_padded, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(2)  # Shape: (N, kernel_h*kernel_w, 1, H_out, W_out)
        sampled = sampled * mask

    # Reshape sampled input for convolution
    sampled = sampled.view(N, C_in, kernel_h, kernel_w, H_out, W_out)
    sampled = sampled.permute(0, 4, 5, 1, 2, 3).contiguous()
    sampled = sampled.view(N * H_out * W_out, C_in * kernel_h * kernel_w)

    # Reshape weight for convolution
    weight = weight.view(C_out, C_in * kernel_h * kernel_w)

    # Perform convolution
    output = torch.mm(sampled, weight.t())
    output = output.view(N, H_out, W_out, C_out)
    output = output.permute(0, 3, 1, 2).contiguous()

    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

