import torch
import torch.nn.functional as F

def spatial_gradient(input_image, mode='sobel', order=1, normalized=False):
    """
    Computes the first-order image derivative in both x and y using a Sobel operator or simple difference.
    
    Parameters:
    - input_image: torch.Tensor of shape (B, C, H, W)
    - mode: str, either 'sobel' or 'diff'
    - order: int, the order of the derivatives (only 1 is supported in this implementation)
    - normalized: bool, whether to normalize the output
    
    Returns:
    - torch.Tensor of shape (B, C, 2, H, W) containing the derivatives
    """
    if order != 1:
        raise ValueError("Only first-order derivatives are supported in this implementation.")
    
    if mode not in ['sobel', 'diff']:
        raise ValueError("Mode must be either 'sobel' or 'diff'.")
    
    B, C, H, W = input_image.shape
    
    if mode == 'sobel':
        # Sobel kernels
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        
        # Apply Sobel filter
        grad_x = F.conv2d(input_image.view(B * C, 1, H, W), sobel_x, padding=1)
        grad_y = F.conv2d(input_image.view(B * C, 1, H, W), sobel_y, padding=1)
    
    elif mode == 'diff':
        # Simple difference kernels
        diff_x = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        diff_y = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32, device=input_image.device).view(1, 1, 3, 3)
        
        # Apply difference filter
        grad_x = F.conv2d(input_image.view(B * C, 1, H, W), diff_x, padding=1)
        grad_y = F.conv2d(input_image.view(B * C, 1, H, W), diff_y, padding=1)
    
    # Reshape to (B, C, 2, H, W)
    grad_x = grad_x.view(B, C, H, W)
    grad_y = grad_y.view(B, C, H, W)
    gradients = torch.stack((grad_x, grad_y), dim=2)
    
    if normalized:
        # Normalize the gradients
        gradients = gradients / (gradients.abs().max(dim=(2, 3, 4), keepdim=True)[0] + 1e-6)
    
    return gradients

