import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

def perspective(img, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR, fill=0):
    if isinstance(img, Image.Image):
        # Convert startpoints and endpoints to numpy arrays
        startpoints = np.array(startpoints, dtype=np.float32)
        endpoints = np.array(endpoints, dtype=np.float32)
        
        # Calculate the perspective transform matrix
        matrix = Image.transform.getperspective(startpoints, endpoints)
        
        # Apply the perspective transform
        return img.transform(img.size, Image.PERSPECTIVE, matrix, resample=interpolation, fillcolor=fill)
    
    elif isinstance(img, torch.Tensor):
        if img.ndim < 2:
            raise ValueError("Tensor image should have at least 2 dimensions")
        
        # Convert startpoints and endpoints to tensors
        startpoints = torch.tensor(startpoints, dtype=torch.float32)
        endpoints = torch.tensor(endpoints, dtype=torch.float32)
        
        # Calculate the perspective transform matrix
        matrix = F._get_perspective_coeffs(startpoints, endpoints)
        
        # Apply the perspective transform
        return F.perspective(img, matrix, interpolation=interpolation, fill=fill)
    
    else:
        raise TypeError("img should be PIL Image or Tensor")

