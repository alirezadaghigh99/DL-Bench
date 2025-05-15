from PIL import Image
import numpy as np
import torch

def to_pil_image(pic, mode=None):
    # Check if the input is a numpy array or a PyTorch tensor
    if not isinstance(pic, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
    
    # Convert PyTorch tensor to numpy array if necessary
    if isinstance(pic, torch.Tensor):
        pic = pic.detach().cpu().numpy()
    
    # Check the dimensions of the input
    if pic.ndim not in {2, 3}:
        raise ValueError("Input image must be 2D or 3D.")
    
    # Handle 2D images (single channel)
    if pic.ndim == 2:
        if mode is None:
            mode = 'L'
        if mode not in {'L', 'I', 'I;16', 'F'}:
            raise ValueError(f"Mode {mode} is not supported for 1-channel images.")
        return Image.fromarray(pic, mode)
    
    # Handle 3D images
    if pic.ndim == 3:
        channels = pic.shape[2]
        if channels > 4:
            raise ValueError("Input image must have at most 4 channels.")
        
        # Default mode based on the number of channels
        if mode is None:
            if channels == 1:
                mode = 'L'
            elif channels == 2:
                mode = 'LA'
            elif channels == 3:
                mode = 'RGB'
            elif channels == 4:
                mode = 'RGBA'
        
        # Validate mode based on the number of channels
        valid_modes = {
            1: {'L', 'I', 'I;16', 'F'},
            2: {'LA'},
            3: {'RGB', 'YCbCr', 'HSV'},
            4: {'RGBA', 'CMYK', 'RGBX'}
        }
        
        if mode not in valid_modes.get(channels, set()):
            raise ValueError(f"Mode {mode} is not supported for {channels}-channel images.")
        
        return Image.fromarray(pic, mode)
    
    # If the function reaches this point, something unexpected happened
    raise TypeError("Unexpected input type or dimensions.")

