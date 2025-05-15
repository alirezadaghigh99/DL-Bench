import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import pathlib

def save_image(tensor, fp, format=None, **kwargs):
    # Check if the input is a list of tensors
    if isinstance(tensor, list):
        # Use make_grid to arrange the list of tensors into a grid
        tensor = make_grid(tensor, **kwargs)
    
    # Ensure the tensor is on the CPU and detach it from the computation graph
    tensor = tensor.cpu().detach()
    
    # Normalize and clamp the tensor to the range [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert the tensor to a NumPy array
    # If the tensor is in the shape (C, H, W), we need to permute it to (H, W, C)
    if tensor.ndimension() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to a NumPy array and scale to [0, 255]
    array = (tensor.numpy() * 255).astype(np.uint8)
    
    # Create a PIL Image from the NumPy array
    image = Image.fromarray(array)
    
    # Determine the format if not provided
    if format is None:
        if isinstance(fp, (str, pathlib.Path)):
            format = pathlib.Path(fp).suffix[1:]  # Get the file extension without the dot
        else:
            raise ValueError("Format must be specified when using a file-like object.")
    
    # Save the image
    image.save(fp, format=format)

