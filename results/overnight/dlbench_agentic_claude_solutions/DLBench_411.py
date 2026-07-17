from __future__ import annotations
from typing import Optional
import torch
from kornia.core import stack
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import transform_points
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
from .distort import distort_points, tilt_projection

def undistort_points(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, new_K: Optional[torch.Tensor]=None, num_iters: int=5) -> torch.Tensor:
    """Compensate for lens distortion a set of 2D image points.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\\tau_x, \\tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\\tau_x,\\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.
        num_iters: Number of undistortion iterations. Default: 5.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 4, 2)
        >>> K = torch.eye(3)[None]
        >>> dist = torch.rand(1, 4)
        >>> undistort_points(x, K, dist)
        tensor([[[-0.1513, -0.1165],
                 [ 0.0711,  0.1100],
                 [-0.0697,  0.0228],
                 [-0.1843, -0.1606]]])
    """
    KORNIA_CHECK_SHAPE(points, ['*', 'N', '2'])
    KORNIA_CHECK_SHAPE(K, ['*', '3', '3'])
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f'points shape is invalid. Got {points.shape}.')
    if new_K is None:
        new_K = K
    else:
        KORNIA_CHECK_SHAPE(new_K, ['*', '3', '3'])
    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f'Invalid number of distortion coefficients. Got {dist.shape[-1]}')
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])
    cx: torch.Tensor = K[..., 0:1, 2]
    cy: torch.Tensor = K[..., 1:2, 2]
    fx: torch.Tensor = K[..., 0:1, 0]
    fy: torch.Tensor = K[..., 1:2, 1]
    x: torch.Tensor = (points[..., 0] - cx) / fx
    y: torch.Tensor = (points[..., 1] - cy) / fy
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)
        (x, y) = transform_points(inv_tilt, stack([x, y], dim=-1)).unbind(-1)
    (x0, y0) = (x, y)
    for _ in range(num_iters):
        r2 = x * x + y * y
        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3) / (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3)
        deltaX = 2 * dist[..., 2:3] * x * y + dist[..., 3:4] * (r2 + 2 * x * x) + dist[..., 8:9] * r2 + dist[..., 9:10] * r2 * r2
        deltaY = dist[..., 2:3] * (r2 + 2 * y * y) + 2 * dist[..., 3:4] * x * y + dist[..., 10:11] * r2 + dist[..., 11:12] * r2 * r2
        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly
    new_cx: torch.Tensor = new_K[..., 0:1, 2]
    new_cy: torch.Tensor = new_K[..., 1:2, 2]
    new_fx: torch.Tensor = new_K[..., 0:1, 0]
    new_fy: torch.Tensor = new_K[..., 1:2, 1]
    x = new_fx * x + new_cx
    y = new_fy * y + new_cy
    return stack([x, y], -1)

def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3, k_4, k_5, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\\tau_x, \\tau_y)`
    distortion models are considered in this function.

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\\tau_x,\\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.

    Returns:
        Undistorted image with shape :math:`(*, C, H, W)`.

    Example:
        >>> img = torch.rand(1, 3, 5, 5)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> out = undistort_image(img, K, dist_coeff)
        >>> out.shape
        torch.Size([1, 3, 5, 5])
    """
    if len(image.shape) < 3:
        raise ValueError(f'Image shape is invalid. Got: {image.shape}.')
    KORNIA_CHECK_SHAPE(K, ['*', '3', '3'])
    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f'Invalid number of distortion coefficients. Got {dist.shape[-1]}')
    rows, cols = (image.shape[-2], image.shape[-1])
    B = image.shape[:-3]
    xy_grid: torch.Tensor = create_meshgrid(rows, cols, normalized_coordinates=False, device=image.device, dtype=image.dtype)
    pts = xy_grid.reshape(-1, 2)
    ptsd = distort_points(pts, K, dist)
    mapx = ptsd[..., 0].reshape(*B, rows, cols)
    mapy = ptsd[..., 1].reshape(*B, rows, cols)
    return remap(image, mapx, mapy, align_corners=True)
