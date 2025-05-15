from PIL import Image
import torchvision.transforms.functional as F
import torch
from torchvision.transforms import InterpolationMode
import warnings

def resize(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
    # Check if interpolation is valid
    if not isinstance(interpolation, (InterpolationMode, int)):
        raise TypeError("Interpolation must be of type InterpolationMode or an integer corresponding to a Pillow constant.")
    
    # Check if size is valid
    if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) in [1, 2])):
        raise ValueError("Size must be an integer or a list/tuple of length 1 or 2.")
    
    # Check if max_size is valid
    if max_size is not None and not isinstance(size, int):
        raise ValueError("max_size can only be used if size is a single integer.")
    
    # Handle PIL Image
    if isinstance(img, Image.Image):
        if not antialias:
            warnings.warn("Antialias is always applied for PIL Images, ignoring the antialias=False setting.")
        
        # Calculate new size maintaining aspect ratio if necessary
        if isinstance(size, int):
            w, h = img.size
            if (max_size is not None) and (max(w, h) > max_size):
                size = (max_size, max_size * h // w) if w > h else (max_size * w // h, max_size)
            else:
                size = (size, size * h // w) if w > h else (size * w // h, size)
        
        return img.resize(size, resample=interpolation)
    
    # Handle Tensor
    elif isinstance(img, torch.Tensor):
        if isinstance(size, int):
            _, h, w = img.shape
            if (max_size is not None) and (max(w, h) > max_size):
                size = (max_size, max_size * h // w) if w > h else (max_size * w // h, max_size)
            else:
                size = (size, size * h // w) if w > h else (size * w // h, size)
        
        return F.resize(img, size, interpolation=interpolation, antialias=antialias)
    
    else:
        raise TypeError("img should be either a PIL Image or a torch Tensor.")

