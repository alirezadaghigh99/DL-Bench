import torch
from typing import Optional, Tuple
import warnings

def preprocess_input(
    image_rgb: Optional[torch.Tensor],
    fg_probability: Optional[torch.Tensor],
    depth_map: Optional[torch.Tensor],
    mask_images: bool,
    mask_depths: bool,
    mask_threshold: float,
    bg_color: Tuple[float, float, float]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    
    # Check if image_rgb is batched correctly
    if image_rgb is not None and (len(image_rgb.shape) != 4 or image_rgb.shape[1] != 3):
        raise ValueError("image_rgb must have shape (B, 3, H, W)")

    # Initialize fg_mask
    fg_mask = None

    # Process foreground probability maps
    if fg_probability is not None:
        if len(fg_probability.shape) != 4 or fg_probability.shape[1] != 1:
            raise ValueError("fg_probability must have shape (B, 1, H, W)")
        
        # Threshold the foreground probability to create a binary mask
        fg_mask = (fg_probability > mask_threshold).float()
        warnings.warn("Thresholding foreground probability maps to create binary masks.")

    # Mask the RGB images
    if image_rgb is not None and mask_images:
        if fg_mask is None:
            raise ValueError("Foreground mask is required to mask images.")
        
        # Expand fg_mask to match the RGB channels
        fg_mask_rgb = fg_mask.expand_as(image_rgb)
        
        # Apply the mask to the images
        image_rgb = image_rgb * fg_mask_rgb + (1 - fg_mask_rgb) * torch.tensor(bg_color, device=image_rgb.device).view(1, 3, 1, 1)
        warnings.warn("Masking RGB images with the foreground mask.")

    # Mask the depth maps
    if depth_map is not None and mask_depths:
        if len(depth_map.shape) != 4 or depth_map.shape[1] != 1:
            raise ValueError("depth_map must have shape (B, 1, H, W)")
        
        if fg_mask is None:
            raise ValueError("Foreground mask is required to mask depth maps.")
        
        # Apply the mask to the depth maps
        depth_map = depth_map * fg_mask + (1 - fg_mask) * torch.tensor(0.0, device=depth_map.device)
        warnings.warn("Masking depth maps with the foreground mask.")

    return image_rgb, fg_mask, depth_map