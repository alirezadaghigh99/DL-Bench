from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def _get_perspective_coeffs(startpoints, endpoints):
    """Calculate coefficients for perspective transformation."""
    matrix = []
    for (x0, y0), (x1, y1) in zip(startpoints, endpoints):
        matrix.append([x0, y0, 1, 0, 0, 0, -x1 * x0, -x1 * y0])
        matrix.append([0, 0, 0, x0, y0, 1, -y1 * x0, -y1 * y0])

    A = np.array(matrix, dtype=np.float32)
    B = np.array(endpoints, dtype=np.float32).reshape(8)

    res = np.linalg.solve(A, B)
    return res.reshape(8)

def perspective(image, startpoints, endpoints, interpolation=Image.BILINEAR, fill=None):
    """Apply a perspective transformation to an image."""
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError("startpoints and endpoints must each contain exactly four points.")

    if isinstance(image, Image.Image):
        coeffs = _get_perspective_coeffs(startpoints, endpoints)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, interpolation, fillcolor=fill)

    elif isinstance(image, torch.Tensor):
        if image.ndimension() != 3:
            raise ValueError("Tensor image should be 3-dimensional (C, H, W).")

        # Convert interpolation to InterpolationMode
        if interpolation == Image.NEAREST:
            interpolation_mode = InterpolationMode.NEAREST
        elif interpolation == Image.BILINEAR:
            interpolation_mode = InterpolationMode.BILINEAR
        elif interpolation == Image.BICUBIC:
            interpolation_mode = InterpolationMode.BICUBIC
        else:
            raise ValueError("Unsupported interpolation mode.")

        # Convert startpoints and endpoints to the format expected by torchvision
        startpoints = [tuple(map(float, pt)) for pt in startpoints]
        endpoints = [tuple(map(float, pt)) for pt in endpoints]

        return F.perspective(image, startpoints, endpoints, interpolation_mode, fill)

    else:
        raise TypeError("Input image must be a PIL Image or a PyTorch Tensor.")

