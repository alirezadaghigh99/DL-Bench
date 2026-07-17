from typing import Union
import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import functional as _F

@torch.jit.unused
def to_image(inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> tv_tensors.Image:
    """Generate a Python function called to_image that takes in an input of type Union[torch.Tensor, PIL.Image.Image, np.ndarray] and returns an output of type tv_tensors.Image. The function first checks the type of the input and converts it accordingly - if the input is a numpy array, it converts it to a torch tensor with at least 3 dimensions and permutes the dimensions to (2, 0, 1). If the input is a PIL image, it uses the pil_to_tensor function to convert it. If the input is already a torch tensor, it returns the input as is. If the input is none of these types, it raises a TypeError. The output is a tv_tensors.Image object."""
    if isinstance(inpt, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(inpt)).permute((2, 0, 1)).contiguous()
    elif isinstance(inpt, PIL.Image.Image):
        output = pil_to_tensor(inpt)
    elif isinstance(inpt, torch.Tensor):
        output = inpt
    else:
        raise TypeError(f"Input can either be a numpy array or a PIL image, but got {type(inpt)} instead.")
    return tv_tensors.Image(output)
to_pil_image = _F.to_pil_image
pil_to_tensor = _F.pil_to_tensor
