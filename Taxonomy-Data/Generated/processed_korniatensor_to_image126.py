import torch
import numpy as np

def tensor_to_image(tensor, keepdim=True, force_contiguous=False):
    # Check if the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Check the number of dimensions
    if tensor.dim() not in {2, 3, 4}:
        raise ValueError("Input tensor must have 2, 3, or 4 dimensions.")
    
    # Move tensor to CPU if it's on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Make tensor contiguous if required
    if force_contiguous:
        tensor = tensor.contiguous()
    
    # Convert tensor to NumPy array
    array = tensor.numpy()
    
    # Handle different shapes
    if tensor.dim() == 2:
        # Shape (H, W)
        return array
    elif tensor.dim() == 3:
        # Shape (C, H, W) -> (H, W, C)
        array = np.transpose(array, (1, 2, 0))
        return array
    elif tensor.dim() == 4:
        # Shape (B, C, H, W) -> (B, H, W, C)
        array = np.transpose(array, (0, 2, 3, 1))
        if not keepdim:
            # Remove batch dimension if keepdim is False and batch size is 1
            if array.shape[0] == 1:
                array = array.squeeze(0)
        return array

