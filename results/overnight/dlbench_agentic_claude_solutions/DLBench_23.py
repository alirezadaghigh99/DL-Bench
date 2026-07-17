import itertools
from typing import Tuple
import torch
import torch.nn.functional as F
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from .blending import _get_background_color

def _precompute(input_shape: Tuple[int, int, int, int], device: Device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute padding and offset constants that won't change for a given NHWK shape.

    Args:
        input_shape: Tuple indicating N (batch size), H, W (image size) and K (number of
            intersections) output by the rasterizer.
        device: Device to store the tensors on.

    returns:
        crop_ids_h: An (N, H, W+2, K, 9, 5) tensor, used during splatting to offset the
            p-pixels (splatting pixels) in one of the 9 splatting directions within a
            call to torch.gather. See comments and offset_splats for details.
        crop_ids_w: An (N, H, W, K, 9, 5) tensor, used similarly to crop_ids_h.
        offsets: A (1, 1, 1, 1, 9, 2) tensor (shaped so for broadcasting) containing va-
            lues [-1, -1], [-1, 0], [-1, 1], [0, -1], ..., [1, 1] which correspond to
            the nine splatting directions.
    """
    (N, H, W, K) = input_shape
    crop_ids_h = (torch.arange(0, H, device=device).view(1, H, 1, 1, 1, 1) + torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device).view(1, 1, 1, 1, 9, 1)).expand(N, H, W + 2, K, 9, 5)
    crop_ids_w = (torch.arange(0, W, device=device).view(1, 1, W, 1, 1, 1) + torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=device).view(1, 1, 1, 1, 9, 1)).expand(N, H, W, K, 9, 5)
    offsets = torch.tensor(list(itertools.product((-1, 0, 1), repeat=2)), dtype=torch.long, device=device)
    return (crop_ids_h, crop_ids_w, offsets)

def _prepare_pixels_and_colors(pixel_coords_cameras: torch.Tensor, colors: torch.Tensor, cameras: FoVPerspectiveCameras, background_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project pixel coords into the un-inverted screen frame of reference, and set
    background pixel z-values to 1.0 and alphas to 0.0.

    Args:
        pixel_coords_cameras: (N, H, W, K, 3) float tensor.
        colors: (N, H, W, K, 3) float tensor.
        cameras: PyTorch3D cameras, for now we assume FoVPerspectiveCameras.
        background_mask: (N, H, W, K) boolean tensor.

    Returns:
        pixel_coords_screen: (N, H, W, K, 3) float tensor. Background pixels have
            x=y=z=1.0.
        colors: (N, H, W, K, 4). Alpha is set to 1 for foreground pixels and 0 for back-
            ground pixels.
    """
    (N, H, W, K, C) = colors.shape
    pixel_coords_screen = cameras.transform_points_screen(pixel_coords_cameras.view([N, -1, 3]), image_size=(H, W), with_xyflip=False).reshape(pixel_coords_cameras.shape)
    colors = torch.cat([colors, torch.ones_like(colors[..., :1])], dim=-1)
    pixel_coords_screen[background_mask] = 1.0
    colors[background_mask] = 0.0
    return (pixel_coords_screen, colors)

def _get_splat_kernel_normalization(offsets: torch.Tensor, sigma: float=0.5):
    if sigma <= 0.0:
        raise ValueError('Only positive standard deviations make sense.')
    epsilon = 0.05
    normalization_constant = torch.exp(-(offsets ** 2).sum(dim=1) / (2 * sigma ** 2)).sum()
    return (1 + epsilon) / normalization_constant

def _compute_occlusion_layers(q_depth: torch.Tensor) -> torch.Tensor:
    """
    For each splatting pixel, decides whether it splats from a background, surface, or
    foreground depth relative to the splatted pixel. See unit tests in
    test_splatter_blend.py for a full example and behavior in ambiguous cases.

    Args:
        q_depth: (N, H, W, K) tensor of per-pixel z-values of the K nearest points at
            each pixel, sorted in ascending order (index 0 is the nearest point).

    Returns:
        occlusion_layers: (N, H, W, 9) long tensor. Each of the 9 values in the last
            dimension corresponds to one of the nine splatting directions ([-1, -1],
            [-1, 0], ..., [1, 1], in the same order as the offsets used elsewhere in
            this file). The value at each direction is compared, elsewhere, against the
            depth-layer index k (0, ..., K-1) of the pixel splatting in that direction:
            layers with k less than this value are in the foreground (nearer to the
            camera than the splatted-to pixel's own surface), k equal to this value are
            on the same surface, and k greater than this value are in the background
            (occluded by the splatted-to pixel's own surface).
    """
    (N, H, W, K) = q_depth.shape
    q_depth_padded = F.pad(q_depth, [0, 0, 1, 1, 1, 1], value=1.0)
    q_surface_depth = q_depth[..., 0:1]
    occlusion_layers = []
    for (dx, dy) in itertools.product((-1, 0, 1), repeat=2):
        p_depth = q_depth_padded[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W, :]
        occlusion_layers.append((p_depth < q_surface_depth).sum(dim=-1))
    return torch.stack(occlusion_layers, dim=-1)

def _compute_splatting_colors_and_weights(pixel_coords_screen: torch.Tensor, colors: torch.Tensor, sigma: float, offsets: torch.Tensor) -> torch.Tensor:
    """
    For each center pixel q, compute the splatting weights of its surrounding nine spla-
    tting pixels p, as well as their splatting colors (which are just their colors re-
    weighted by the splatting weights).

    Args:
        pixel_coords_screen: (N, H, W, K, 2) tensor of pixel screen coords.
        colors: (N, H, W, K, 4) RGBA tensor of pixel colors.
        sigma: splatting kernel variance.
        offsets: (9, 2) tensor computed by _precompute, indicating the nine
            splatting directions ([-1, -1], ..., [1, 1]).

    Returns:
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor.
            splat_colors_and_weights[..., :4] corresponds to the splatting colors, and
            splat_colors_and_weights[..., 4:5] to the splatting weights. The "9" di-
            mension corresponds to the nine splatting directions.
    """
    (N, H, W, K, C) = colors.shape
    splat_kernel_normalization = _get_splat_kernel_normalization(offsets, sigma)
    q_to_px_center = (torch.floor(pixel_coords_screen[..., :2]) - pixel_coords_screen[..., :2] + 0.5).view((N, H, W, K, 1, 2))
    dist2_p_q = torch.sum((q_to_px_center + offsets) ** 2, dim=5)
    splat_weights = torch.exp(-dist2_p_q / (2 * sigma ** 2))
    alpha = colors[..., 3:4]
    splat_weights = (alpha * splat_kernel_normalization * splat_weights).unsqueeze(5)
    splat_colors = splat_weights * colors.unsqueeze(4)
    return torch.cat([splat_colors, splat_weights], dim=5)

def _offset_splats(splat_colors_and_weights: torch.Tensor, crop_ids_h: torch.Tensor, crop_ids_w: torch.Tensor) -> torch.Tensor:
    """
    Pad splatting colors and weights so that tensor locations/coordinates are aligned
    with the splatting directions. For example, say we have an example input Red channel
    splat_colors_and_weights[n, :, :, k, direction=0, channel=0] equal to
       .1  .2  .3
       .4  .5  .6
       .7  .8  .9
    the (h, w) entry indicates that pixel n, h, w, k splats the given color in direction
    equal to 0, which corresponds to offsets[0] = (-1, -1). Note that this is the x-y
    direction, not h-w. This function pads and crops this array to
        0   0   0
       .2  .3   0
       .5  .6   0
    which indicates, for example, that:
        * There is no pixel splatting in direction (-1, -1) whose splat lands on pixel
          h=w=0.
        * There is a pixel splatting in direction (-1, -1) whose splat lands on the pi-
          xel h=1, w=0, and that pixel's splatting color is .2.
        * There is a pixel splatting in direction (-1, -1) whose splat lands on the pi-
          xel h=2, w=1, and that pixel's splatting color is .6.

    Args:
        *splat_colors_and_weights*: (N, H, W, K, 9, 5) tensor of colors and weights,
        where dim=-2 corresponds to the splatting directions/offsets.
        *crop_ids_h*: (N, H, W+2, K, 9, 5) precomputed tensor used for padding within
            torch.gather. See _precompute for more info.
        *crop_ids_w*: (N, H, W, K, 9, 5) precomputed tensor used for padding within
            torch.gather. See _precompute for more info.


    Returns:
        *splat_colors_and_weights*: (N, H, W, K, 9, 5) tensor.
    """
    (N, H, W, K, _, _) = splat_colors_and_weights.shape
    splat_colors_and_weights = F.pad(splat_colors_and_weights, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    splat_colors_and_weights = torch.gather(splat_colors_and_weights, dim=1, index=crop_ids_h)
    splat_colors_and_weights = torch.gather(splat_colors_and_weights, dim=2, index=crop_ids_w)
    return splat_colors_and_weights

def _compute_splatted_colors_and_weights(occlusion_layers: torch.Tensor, splat_colors_and_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate splatted colors in background, surface and foreground occlusion buffers.

    Args:
        occlusion_layers: (N, H, W, 9) tensor. See _compute_occlusion_layers.
        splat_colors_and_weights: (N, H, W, K, 9, 5) tensor. See _offset_splats.

    Returns:
        splatted_colors: (N, H, W, 4, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat colors.
        splatted_weights: (N, H, W, 1, 3) tensor. Last dimension corresponds to back-
            ground, surface, and foreground splat weights and is used for normalization.

    """
    (N, H, W, K, _, _) = splat_colors_and_weights.shape
    layer_ids = torch.arange(K, device=splat_colors_and_weights.device).view(1, 1, 1, K, 1)
    occlusion_layers = occlusion_layers.view(N, H, W, 1, 9)
    occlusion_layer_mask = torch.stack([occlusion_layers > layer_ids, occlusion_layers == layer_ids, occlusion_layers < layer_ids], dim=5).float()
    splatted_colors_and_weights = torch.bmm(splat_colors_and_weights.permute(0, 1, 2, 5, 3, 4).reshape((N * H * W, 5, K * 9)), occlusion_layer_mask.reshape((N * H * W, K * 9, 3))).reshape((N, H, W, 5, 3))
    return (splatted_colors_and_weights[..., :4, :], splatted_colors_and_weights[..., 4:5, :])

def _normalize_and_compose_all_layers(background_color: torch.Tensor, splatted_colors_per_occlusion_layer: torch.Tensor, splatted_weights_per_occlusion_layer: torch.Tensor) -> torch.Tensor:
    """
    Normalize each bg/surface/fg buffer by its weight, and compose.

    Args:
        background_color: (3) RGB tensor.
        splatter_colors_per_occlusion_layer: (N, H, W, 4, 3) RGBA tensor, last dimension
            corresponds to foreground, surface, and background splatting.
        splatted_weights_per_occlusion_layer: (N, H, W, 1, 3) weight tensor.

    Returns:
        output_colors: (N, H, W, 4) RGBA tensor.
    """
    device = splatted_colors_per_occlusion_layer.device
    normalization_scales = 1.0 / torch.maximum(splatted_weights_per_occlusion_layer, torch.tensor([1.0], device=device))
    normalized_splatted_colors = splatted_colors_per_occlusion_layer * normalization_scales
    output_colors = torch.cat([background_color, torch.tensor([0.0], device=device)])
    for occlusion_layer_id in (-1, -2, -3):
        alpha = normalized_splatted_colors[..., 3:4, occlusion_layer_id]
        output_colors = normalized_splatted_colors[..., occlusion_layer_id] + (1.0 - alpha) * output_colors
    return output_colors

class SplatterBlender(torch.nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int, int], device):
        """
        A splatting blender. See `forward` docs for details of the splatting mechanism.

        Args:
            input_shape: Tuple (N, H, W, K) indicating the batch size, image height,
                image width, and number of rasterized layers. Used to precompute
                constant tensors that do not change as long as this tuple is unchanged.
        """
        super().__init__()
        (self.crop_ids_h, self.crop_ids_w, self.offsets) = _precompute(input_shape, device)

    def to(self, device):
        self.offsets = self.offsets.to(device)
        self.crop_ids_h = self.crop_ids_h.to(device)
        self.crop_ids_w = self.crop_ids_w.to(device)
        super().to(device)

    def forward(self, colors: torch.Tensor, pixel_coords_cameras: torch.Tensor, cameras: FoVPerspectiveCameras, background_mask: torch.Tensor, blend_params: BlendParams) -> torch.Tensor:
        """
        RGB blending using splatting, as proposed in [0].

        Args:
            colors: (N, H, W, K, 3) tensor of RGB colors at each h, w pixel location for
                K intersection layers.
            pixel_coords_cameras: (N, H, W, K, 3) tensor of pixel coordinates in the
                camera frame of reference. It is *crucial* that these are computed by
                interpolating triangle vertex positions using barycentric coordinates --
                this allows gradients to travel through pixel_coords_camera back to the
                vertex positions.
            cameras: Cameras object used to project pixel_coords_cameras screen coords.
            background_mask: (N, H, W, K, 3) boolean tensor, True for bg pixels. A pixel
                is considered "background" if no mesh triangle projects to it. This is
                typically computed by the rasterizer.
            blend_params: BlendParams, from which we use sigma (splatting kernel
                variance) and background_color.

        Returns:
            output_colors: (N, H, W, 4) tensor of RGBA values. The alpha layer is set to
                fully transparent in the background.

        [0] Cole, F. et al., "Differentiable Surface Rendering via Non-differentiable
            Sampling".
        """
        (pixel_coords_screen, colors) = _prepare_pixels_and_colors(pixel_coords_cameras, colors, cameras, background_mask)
        occlusion_layers = _compute_occlusion_layers(pixel_coords_screen[..., 2:3].squeeze(dim=-1))
        splat_colors_and_weights = _compute_splatting_colors_and_weights(pixel_coords_screen, colors, blend_params.sigma, self.offsets)
        splat_colors_and_weights = _offset_splats(splat_colors_and_weights, self.crop_ids_h, self.crop_ids_w)
        (splatted_colors_per_occlusion_layer, splatted_weights_per_occlusion_layer) = _compute_splatted_colors_and_weights(occlusion_layers, splat_colors_and_weights)
        output_colors = _normalize_and_compose_all_layers(_get_background_color(blend_params, colors.device), splatted_colors_per_occlusion_layer, splatted_weights_per_occlusion_layer)
        return output_colors
