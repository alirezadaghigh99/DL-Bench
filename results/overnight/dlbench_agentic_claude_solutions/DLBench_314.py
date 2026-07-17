from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale
from kornia.core import ImageModule as Module
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from .gaussian import gaussian_blur2d
from .kernels import get_canny_nms_kernel, get_hysteresis_kernel
from .sobel import spatial_gradient

def canny(input: Tensor, low_threshold: float=0.1, high_threshold: float=0.2, kernel_size: tuple[int, int] | int=(5, 5), sigma: tuple[float, float] | Tensor=(1, 1), hysteresis: bool=True, eps: float=1e-06) -> tuple[Tensor, Tensor]:
    """Generate a Python function called canny that implements the Canny edge detection algorithm. The function takes the following inputs:
- input: input image tensor with shape (B,C,H,W)
- low_threshold: lower threshold for the hysteresis procedure
- high_threshold: upper threshold for the hysteresis procedure
- kernel_size: the size of the kernel for the Gaussian blur
- sigma: the standard deviation of the kernel for the Gaussian blur
- hysteresis: a boolean indicating whether to apply hysteresis edge tracking
- eps: a regularization number to avoid NaN during backpropagation

The function returns a tuple containing:
- the canny edge magnitudes map, with a shape of (B,1,H,W)
- the canny edge detection filtered by thresholds and hysteresis, with a shape of (B,1,H,W)

The function first checks the input tensor and its shape, then converts the input to grayscale if it has 3 channels. It applies Gaussian blur, computes gradients, computes gradient magnitude and angle, performs non-maximal suppression, applies thresholding, and finally applies hysteresis if specified. The output edges are returned as tensors."""
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    KORNIA_CHECK(low_threshold <= high_threshold, f'Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: {low_threshold}>{high_threshold}')
    KORNIA_CHECK(0 < low_threshold < 1, f'Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}')
    KORNIA_CHECK(0 < high_threshold < 1, f'Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}')

    device = input.device
    dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter noise reduction
    blurred: Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients: Tensor = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx: Tensor = gradients[:, :, 0]
    gy: Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude: Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: Tensor = torch.atan2(gy, gx)

    # Radians to degrees
    angle = 180.0 * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels: Tensor = get_canny_nms_kernel(device, dtype)
    nms_magnitude: Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx: Tensor = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx: Tensor = (positive_idx + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: Tensor = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative], 1)

    is_max: Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

    magnitude = magnitude * is_max

    # Threshold
    low: Tensor = magnitude > low_threshold
    high: Tensor = magnitude > high_threshold

    edges: Tensor = (low * 0.5 + high * 0.5).to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old: Tensor = -torch.ones(edges.shape, device=device, dtype=dtype)
        hysteresis_kernels: Tensor = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            edges_old = edges.clone()

            weak: Tensor = (edges == 0.5).to(dtype)
            strong: Tensor = (edges == 1).to(dtype)

            neighbor_vals: Tensor = F.conv2d(edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2)
            has_strong_neighbor: Tensor = (neighbor_vals == 1).any(dim=1, keepdim=True).to(dtype)

            edges = strong + weak * has_strong_neighbor + weak * (1.0 - has_strong_neighbor) * 0.5

        edges = (edges == 1.0).to(dtype)

    return magnitude, edges

class Canny(Module):
    """Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    ONNX_EXPORTABLE = False

    def __init__(self, low_threshold: float=0.1, high_threshold: float=0.2, kernel_size: tuple[int, int] | int=(5, 5), sigma: tuple[float, float] | Tensor=(1, 1), hysteresis: bool=True, eps: float=1e-06) -> None:
        super().__init__()
        KORNIA_CHECK(low_threshold <= high_threshold, f'Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: {low_threshold}>{high_threshold}')
        KORNIA_CHECK(0 < low_threshold < 1, f'Invalid low threshold. Should be in range (0, 1). Got: {low_threshold}')
        KORNIA_CHECK(0 < high_threshold < 1, f'Invalid high threshold. Should be in range (0, 1). Got: {high_threshold}')
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hysteresis = hysteresis
        self.eps: float = eps

    def __repr__(self) -> str:
        return ''.join((f'{type(self).__name__}(', ', '.join((f'{name}={getattr(self, name)}' for name in sorted(self.__dict__) if not name.startswith('_'))), ')'))

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        return canny(input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps)
