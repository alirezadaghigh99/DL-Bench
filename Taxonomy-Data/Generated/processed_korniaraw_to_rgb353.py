import torch
import torch.nn.functional as F

def raw_to_rgb(image: torch.Tensor, cfa: str) -> torch.Tensor:
    # Validate input type
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}.")
    
    # Validate input dimensions
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")
    
    # Validate height and width divisibility by 2
    _, _, H, W = image.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Input H&W must be evenly divisible by 2. Got {image.shape}.")
    
    # Define kernels for bilinear interpolation
    kernel_red_blue = torch.tensor([[0.25, 0.5, 0.25],
                                    [0.5,  1.0, 0.5],
                                    [0.25, 0.5, 0.25]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    kernel_green = torch.tensor([[0, 0.25, 0],
                                 [0.25, 1.0, 0.25],
                                 [0, 0.25, 0]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # Initialize RGB channels
    R = torch.zeros_like(image)
    G = torch.zeros_like(image)
    B = torch.zeros_like(image)
    
    # Handle different CFA configurations
    if cfa == 'BG':
        B[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        G[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        G[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        R[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'GB':
        G[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        B[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        R[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        G[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'RG':
        R[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        G[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        G[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        B[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == 'GR':
        G[:, :, 0::2, 0::2] = image[:, :, 0::2, 0::2]
        R[:, :, 0::2, 1::2] = image[:, :, 0::2, 1::2]
        B[:, :, 1::2, 0::2] = image[:, :, 1::2, 0::2]
        G[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    else:
        raise ValueError(f"Unsupported CFA configuration: {cfa}")
    
    # Apply bilinear interpolation
    R = F.conv2d(R, kernel_red_blue, padding=1)
    B = F.conv2d(B, kernel_red_blue, padding=1)
    G = F.conv2d(G, kernel_green, padding=1)
    
    # Concatenate the channels to form the RGB image
    rgb_image = torch.cat((R, G, B), dim=1)
    
    return rgb_image