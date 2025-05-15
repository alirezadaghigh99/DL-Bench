import torch

def get_sobel_kernel2d(device=None, dtype=None):
    # Define the Sobel kernels for x and y directions
    sobel_kernel_x = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=dtype, device=device)
    
    sobel_kernel_y = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=dtype, device=device)
    
    # Stack the kernels to create a 2D tensor
    sobel_kernels = torch.stack([sobel_kernel_x, sobel_kernel_y])
    
    return sobel_kernels

