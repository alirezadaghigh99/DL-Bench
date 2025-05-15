from typing import Union
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import tv_tensors
from PIL import Image
import numpy as np

def to_image(input: Union[Tensor, Image.Image, np.ndarray]) -> tv_tensors.Image:
    if isinstance(input, np.ndarray):
        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(input)
        # Ensure the tensor has at least 3 dimensions
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)  # Change from HWC to CHW
        else:
            raise ValueError("Unsupported numpy array shape for image conversion.")
        return tv_tensors.Image(tensor)
    
    elif isinstance(input, Image.Image):
        # Convert PIL image to torch tensor
        tensor = F.pil_to_tensor(input)
        return tv_tensors.Image(tensor)
    
    elif isinstance(input, Tensor):
        # Assume the tensor is already in the correct format
        return tv_tensors.Image(input)
    
    else:
        raise TypeError("Input must be a torch.Tensor, PIL.Image.Image, or np.ndarray.")

