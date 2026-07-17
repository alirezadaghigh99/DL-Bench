from __future__ import annotations
import math
from typing import ClassVar
import torch
from kornia.core import ImageModule as Module
from kornia.core import Tensor, stack, tensor, where

def rgb_to_hls(image: Tensor, eps: float=1e-08) -> Tensor:
    """Generate a Python function called rgb_to_hls that converts an RGB image to HLS format. The function takes in a PyTorch tensor called image representing the RGB image with shape (*, 3, H, W) and an epsilon value eps to avoid division by zero. The image data is assumed to be in the range of (0, 1).

The function returns the HLS version of the input image with the same shape (*, 3, H, W). If the input image is not a PyTorch tensor, a TypeError is raised. If the input size does not have a shape of (*, 3, H, W), a ValueError is raised.

The conversion process involves calculating the HLS components (hue, luminance, saturation) based on the RGB values of the input image. The resulting HLS image is returned as a PyTorch tensor.

An example usage of the rgb_to_hls function is provided in the code snippet, where a random input RGB image tensor is converted to HLS format. Raise the value error if there input not instance of image or there was a problem with shape"""
    if not isinstance(image, Tensor):
        raise TypeError(f'Input type is not a Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    maxc, _ = image.max(-3)
    minc, _ = image.min(-3)
    l_ch = (maxc + minc) / 2.0
    deltac = maxc - minc
    s_ch = where(l_ch < 0.5, deltac / (maxc + minc + eps), deltac / (2.0 - maxc - minc + eps))
    s_ch = where(deltac == 0, torch.zeros_like(s_ch), s_ch)
    deltac_safe = where(deltac == 0, torch.ones_like(deltac), deltac)
    hr = (g - b) / (deltac_safe + eps)
    hg = (b - r) / (deltac_safe + eps) + 2.0
    hb = (r - g) / (deltac_safe + eps) + 4.0
    h_ch = where(maxc == r, hr, where(maxc == g, hg, hb))
    h_ch = where(deltac == 0, torch.zeros_like(h_ch), h_ch)
    h_ch = h_ch * (math.pi / 3.0) % (2 * math.pi)
    return stack([h_ch, l_ch, s_ch], dim=-3)

def hls_to_rgb(image: Tensor) -> Tensor:
    """Convert a HLS image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: HLS image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hls_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f'Input type is not a Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    _HLS2RGB = tensor([[[0.0]], [[8.0]], [[4.0]]], device=image.device, dtype=image.dtype)
    im: Tensor = image.unsqueeze(-4)
    h_ch: Tensor = im[..., 0, :, :]
    l_ch: Tensor = im[..., 1, :, :]
    s_ch: Tensor = im[..., 2, :, :]
    h_ch = h_ch * (6 / math.pi)
    a = s_ch * torch.min(l_ch, 1.0 - l_ch)
    k: Tensor = (h_ch + _HLS2RGB) % 12
    mink = torch.min(k - 3.0, 9.0 - k)
    return torch.addcmul(l_ch, a, mink.clamp_(min=-1.0, max=1.0), value=-1)

class RgbToHls(Module):
    """Convert an image from RGB to HLS.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        HLS version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> hls = RgbToHls()
        >>> output = hls(input)  # 2x3x4x5
    """
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_hls(image)

class HlsToRgb(Module):
    """Convert an image from HLS to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - input: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = HlsToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]

    def forward(self, image: Tensor) -> Tensor:
        return hls_to_rgb(image)
