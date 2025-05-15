from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def rotate(img, angle, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if isinstance(interpolation, int):
        interpolation = InterpolationMode(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(img, Image.Image):
        # Handle PIL Image
        return img.rotate(angle, resample=interpolation.value, expand=expand, center=center, fillcolor=fill)
    elif isinstance(img, torch.Tensor):
        # Handle Tensor
        if interpolation not in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            raise ValueError("For Tensors, only NEAREST and BILINEAR interpolation modes are supported.")
        
        return F.rotate(img, angle, interpolation=interpolation, expand=expand, center=center, fill=fill)
    else:
        raise TypeError("img should be PIL Image or Tensor")

