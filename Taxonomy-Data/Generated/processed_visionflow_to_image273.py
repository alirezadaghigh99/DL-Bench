import torch
import numpy as np

def flow_to_image(flow):
    if not isinstance(flow, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    
    if flow.dtype != torch.float:
        raise ValueError("Input tensor must be of type torch.float.")
    
    if flow.ndim not in {3, 4}:
        raise ValueError("Input tensor must have 3 or 4 dimensions.")
    
    if flow.ndim == 3:
        if flow.shape[0] != 2:
            raise ValueError("For 3D input, the shape must be (2, H, W).")
        flow = flow.unsqueeze(0)  # Add batch dimension for consistency
    
    elif flow.ndim == 4:
        if flow.shape[1] != 2:
            raise ValueError("For 4D input, the shape must be (N, 2, H, W).")
    
    N, _, H, W = flow.shape
    
    # Compute the magnitude and angle of the flow
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    magnitude = torch.sqrt(u ** 2 + v ** 2)
    angle = torch.atan2(v, u)
    
    # Normalize the magnitude
    max_magnitude = torch.max(magnitude, dim=(1, 2), keepdim=True)[0]
    max_magnitude[max_magnitude == 0] = 1  # Avoid division by zero
    magnitude = magnitude / max_magnitude
    
    # Convert angle to hue
    hue = (angle + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1]
    
    # Create HSV image
    hsv = torch.zeros((N, 3, H, W), dtype=torch.float, device=flow.device)
    hsv[:, 0, :, :] = hue  # Hue
    hsv[:, 1, :, :] = 1    # Saturation
    hsv[:, 2, :, :] = magnitude  # Value
    
    # Convert HSV to RGB
    rgb = hsv_to_rgb(hsv)
    
    if flow.ndim == 3:
        rgb = rgb.squeeze(0)  # Remove batch dimension if it was added
    
    return rgb

def hsv_to_rgb(hsv):
    # Convert HSV to RGB using PyTorch
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    z = torch.zeros_like(h)
    
    # Create RGB channels
    rgb = torch.stack([
        torch.where((0 <= h) & (h < 1/6), c, torch.where((1/6 <= h) & (h < 2/6), x, torch.where((2/6 <= h) & (h < 3/6), z, torch.where((3/6 <= h) & (h < 4/6), z, torch.where((4/6 <= h) & (h < 5/6), x, c))))),
        torch.where((0 <= h) & (h < 1/6), x, torch.where((1/6 <= h) & (h < 2/6), c, torch.where((2/6 <= h) & (h < 3/6), c, torch.where((3/6 <= h) & (h < 4/6), x, torch.where((4/6 <= h) & (h < 5/6), z, z))))),
        torch.where((0 <= h) & (h < 1/6), z, torch.where((1/6 <= h) & (h < 2/6), z, torch.where((2/6 <= h) & (h < 3/6), x, torch.where((3/6 <= h) & (h < 4/6), c, torch.where((4/6 <= h) & (h < 5/6), c, x)))))
    ], dim=1)
    
    rgb = rgb + m.unsqueeze(1)
    return rgb