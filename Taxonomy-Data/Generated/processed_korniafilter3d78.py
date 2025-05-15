import torch
import torch.nn.functional as F

def filter3d(input, kernel, border_type='reflect', normalized=False):
    """
    Convolves a 3D kernel with a given input tensor.

    Parameters:
    - input (torch.Tensor): Input tensor of shape (B, C, D, H, W).
    - kernel (torch.Tensor): Kernel tensor of shape (kD, kH, kW).
    - border_type (str): Padding mode ('reflect', 'replicate', 'constant', 'circular').
    - normalized (bool): If True, normalize the kernel using L1 norm.

    Returns:
    - torch.Tensor: Convolved tensor of the same shape as input (B, C, D, H, W).
    """
    # Ensure the kernel is 3D
    assert kernel.dim() == 3, "Kernel must be a 3D tensor"

    # Normalize the kernel if required
    if normalized:
        kernel = kernel / kernel.abs().sum()

    # Get kernel dimensions
    kD, kH, kW = kernel.shape

    # Calculate padding sizes
    pad_d = (kD - 1) // 2
    pad_h = (kH - 1) // 2
    pad_w = (kW - 1) // 2

    # Pad the input tensor
    padded_input = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode=border_type)

    # Expand kernel dimensions to match input channels
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kD, kH, kW)
    kernel = kernel.expand(input.size(1), 1, kD, kH, kW)  # Shape: (C, 1, kD, kH, kW)

    # Perform 3D convolution
    convolved = F.conv3d(padded_input, kernel, groups=input.size(1))

    return convolved

