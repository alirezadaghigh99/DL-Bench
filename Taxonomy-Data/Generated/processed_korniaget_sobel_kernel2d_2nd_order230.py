import torch

def get_sobel_kernel_5x5_2nd_order():
    # Define the 5x5 Sobel kernel for the second derivative in the x-direction (gxx)
    gxx = torch.tensor([
        [1,  4,  6,  4, 1],
        [2,  8, 12,  8, 2],
        [0,  0,  0,  0, 0],
        [-2, -8, -12, -8, -2],
        [-1, -4, -6, -4, -1]
    ], dtype=torch.float32)
    return gxx

def _get_sobel_kernel_5x5_2nd_order_xy():
    # Define the 5x5 Sobel kernel for the mixed partial derivative (gxy)
    gxy = torch.tensor([
        [1,  2,  0, -2, -1],
        [4,  8,  0, -8, -4],
        [6, 12,  0, -12, -6],
        [4,  8,  0, -8, -4],
        [1,  2,  0, -2, -1]
    ], dtype=torch.float32)
    return gxy

def get_sobel_kernel2d_2nd_order(device=None, dtype=None):
    # Get the 5x5 2nd-order Sobel kernel for the second derivative in the x-direction (gxx)
    gxx = get_sobel_kernel_5x5_2nd_order()
    
    # Transpose this kernel to obtain the kernel for the second derivative in the y-direction (gyy)
    gyy = gxx.t()
    
    # Get the mixed partial derivative kernel (gxy)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy()
    
    # Stack the gxx, gxy, and gyy kernels into a single tensor
    kernels = torch.stack([gxx, gxy, gyy], dim=0)
    
    # Move the tensor to the specified device and dtype if provided
    if device is not None:
        kernels = kernels.to(device)
    if dtype is not None:
        kernels = kernels.to(dtype)
    
    return kernels

