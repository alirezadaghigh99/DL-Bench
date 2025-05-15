import torch
import numpy as np
import random

def draw_segmentation_masks(image, masks, alpha=0.5, colors=None):
    # Validate inputs
    if not isinstance(image, torch.Tensor):
        raise TypeError("The image must be a PyTorch tensor.")
    
    if image.dtype not in [torch.uint8, torch.float32, torch.float64]:
        raise ValueError("The image tensor's dtype must be uint8 or a floating point type.")
    
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("The image tensor must have 3 dimensions and be an RGB image (3 channels).")
    
    if masks.ndim not in [2, 3]:
        raise ValueError("The masks tensor must have 2 or 3 dimensions.")
    
    if masks.dtype != torch.bool:
        raise ValueError("The masks tensor must be of boolean dtype.")
    
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)  # Convert to (1, H, W) if single mask
    
    num_masks, H, W = masks.shape
    if (H, W) != image.shape[1:]:
        raise ValueError("The masks tensor's spatial dimensions must match the image tensor.")
    
    # Handle colors
    if colors is None:
        colors = [tuple(np.random.randint(0, 256, size=3)) for _ in range(num_masks)]
    elif isinstance(colors, (list, tuple)) and all(isinstance(c, (list, tuple)) for c in colors):
        if len(colors) != num_masks:
            raise ValueError("The number of colors must match the number of masks.")
    else:
        colors = [colors] * num_masks
    
    # Ensure colors are in the correct format
    colors = [torch.tensor(c, dtype=image.dtype, device=image.device) for c in colors]
    
    # Prepare the output image
    output_image = image.clone()
    
    # Apply each mask
    for i in range(num_masks):
        mask = masks[i]
        color = colors[i]
        
        # Create a colored mask
        colored_mask = torch.zeros_like(image)
        for c in range(3):  # RGB channels
            colored_mask[c] = mask * color[c]
        
        # Blend the mask with the image
        output_image = torch.where(mask, (1 - alpha) * output_image + alpha * colored_mask, output_image)
    
    return output_image

