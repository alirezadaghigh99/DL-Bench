import torch

def rgb_to_hls(image, eps=1e-10):
    # Check if the input is a PyTorch tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check if the input has the correct shape
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("Input must have shape (*, 3, H, W).")
    
    # Split the image into R, G, B components
    r, g, b = image.unbind(dim=-3)
    
    # Calculate max and min values for each pixel
    max_rgb, _ = torch.max(image, dim=-3)
    min_rgb, _ = torch.min(image, dim=-3)
    
    # Calculate Luminance
    l = (max_rgb + min_rgb) / 2
    
    # Calculate Saturation
    s = torch.where(
        l < 0.5,
        (max_rgb - min_rgb) / (max_rgb + min_rgb + eps),
        (max_rgb - min_rgb) / (2.0 - max_rgb - min_rgb + eps)
    )
    
    # Calculate Hue
    delta = max_rgb - min_rgb
    delta = torch.where(delta == 0, torch.tensor(eps, device=image.device), delta)  # Avoid division by zero
    
    hue = torch.zeros_like(l)
    mask = (max_rgb == r)
    hue[mask] = (g[mask] - b[mask]) / delta[mask]
    
    mask = (max_rgb == g)
    hue[mask] = 2.0 + (b[mask] - r[mask]) / delta[mask]
    
    mask = (max_rgb == b)
    hue[mask] = 4.0 + (r[mask] - g[mask]) / delta[mask]
    
    hue = (hue / 6.0) % 1.0  # Normalize hue to be in [0, 1]
    
    # Stack the H, L, S components back together
    hls_image = torch.stack((hue, l, s), dim=-3)
    
    return hls_image

