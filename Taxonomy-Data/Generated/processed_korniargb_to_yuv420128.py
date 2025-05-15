import torch
from typing import Tuple

def rgb_to_yuv420(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if image.dim() < 3 or image.size(-3) != 3:
        raise ValueError("Input image must have shape (*, 3, H, W)")
    
    H, W = image.size(-2), image.size(-1)
    
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("Height and width of the image must be even numbers")
    
    # RGB to YUV conversion matrix
    rgb_to_yuv_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], dtype=image.dtype, device=image.device)
    
    # Reshape image to (N, 3, H, W) for batch processing
    original_shape = image.shape
    image = image.view(-1, 3, H, W)
    
    # Convert RGB to YUV
    yuv_image = torch.tensordot(image.permute(0, 2, 3, 1), rgb_to_yuv_matrix, dims=([3], [1]))
    yuv_image = yuv_image.permute(0, 3, 1, 2)  # (N, 3, H, W)
    
    # Extract Y, U, V channels
    Y = yuv_image[:, 0:1, :, :]  # (N, 1, H, W)
    U = yuv_image[:, 1:2, :, :]  # (N, 1, H, W)
    V = yuv_image[:, 2:3, :, :]  # (N, 1, H, W)
    
    # Subsample U and V channels
    U_subsampled = U[:, :, ::2, ::2]  # (N, 1, H/2, W/2)
    V_subsampled = V[:, :, ::2, ::2]  # (N, 1, H/2, W/2)
    
    # Concatenate U and V channels
    UV = torch.cat((U_subsampled, V_subsampled), dim=1)  # (N, 2, H/2, W/2)
    
    # Reshape back to original batch size
    Y = Y.view(*original_shape[:-3], 1, H, W)
    UV = UV.view(*original_shape[:-3], 2, H // 2, W // 2)
    
    return Y, UV

