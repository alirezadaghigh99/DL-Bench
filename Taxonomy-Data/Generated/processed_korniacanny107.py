import torch
import torch.nn.functional as F
import numpy as np

def canny(input, low_threshold, high_threshold, kernel_size, sigma, hysteresis=True, eps=1e-6):
    # Check input tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if input.dim() != 4:
        raise ValueError("Input tensor must have shape (B, C, H, W)")
    
    B, C, H, W = input.shape
    
    # Convert to grayscale if input has 3 channels
    if C == 3:
        input = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
        input = input.unsqueeze(1)  # Add channel dimension back
    elif C != 1:
        raise ValueError("Input tensor must have 1 or 3 channels")
    
    # Apply Gaussian blur
    def gaussian_kernel(kernel_size, sigma):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel
    
    kernel = gaussian_kernel(kernel_size, sigma).to(input.device)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    blurred = F.conv2d(input, kernel, padding=kernel_size//2, groups=C)
    
    # Compute gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=input.device).expand(C, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=input.device).expand(C, 1, 3, 3)
    
    grad_x = F.conv2d(blurred, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(blurred, sobel_y, padding=1, groups=C)
    
    # Compute gradient magnitude and angle
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    grad_angle = torch.atan2(grad_y, grad_x)
    
    # Non-maximal suppression
    def non_max_suppression(magnitude, angle):
        B, C, H, W = magnitude.shape
        output = torch.zeros_like(magnitude)
        angle = angle * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, H-1):
            for j in range(1, W-1):
                q = 255
                r = 255
                # Angle 0
                if (0 <= angle[:, :, i, j] < 22.5) or (157.5 <= angle[:, :, i, j] <= 180):
                    q = magnitude[:, :, i, j+1]
                    r = magnitude[:, :, i, j-1]
                # Angle 45
                elif 22.5 <= angle[:, :, i, j] < 67.5:
                    q = magnitude[:, :, i+1, j-1]
                    r = magnitude[:, :, i-1, j+1]
                # Angle 90
                elif 67.5 <= angle[:, :, i, j] < 112.5:
                    q = magnitude[:, :, i+1, j]
                    r = magnitude[:, :, i-1, j]
                # Angle 135
                elif 112.5 <= angle[:, :, i, j] < 157.5:
                    q = magnitude[:, :, i-1, j-1]
                    r = magnitude[:, :, i+1, j+1]
                
                if (magnitude[:, :, i, j] >= q) and (magnitude[:, :, i, j] >= r):
                    output[:, :, i, j] = magnitude[:, :, i, j]
                else:
                    output[:, :, i, j] = 0
        return output
    
    suppressed = non_max_suppression(grad_magnitude, grad_angle)
    
    # Apply thresholding
    strong_edges = (suppressed >= high_threshold).float()
    weak_edges = ((suppressed >= low_threshold) & (suppressed < high_threshold)).float()
    
    # Hysteresis
    if hysteresis:
        def hysteresis_tracking(strong, weak):
            B, C, H, W = strong.shape
            edges = strong.clone()
            for i in range(1, H-1):
                for j in range(1, W-1):
                    if weak[:, :, i, j] == 1:
                        if ((strong[:, :, i+1, j-1:j+2] == 1).any() or
                            (strong[:, :, i-1, j-1:j+2] == 1).any() or
                            (strong[:, :, i, [j-1, j+1]] == 1).any()):
                            edges[:, :, i, j] = 1
            return edges
        
        edges = hysteresis_tracking(strong_edges, weak_edges)
    else:
        edges = strong_edges
    
    return grad_magnitude.unsqueeze(1), edges.unsqueeze(1)

