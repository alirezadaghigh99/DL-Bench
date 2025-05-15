def _normalize_and_compose_all_layers(
    background_color: torch.Tensor,
    splatted_colors_per_occlusion_layer: torch.Tensor,
    splatted_weights_per_occlusion_layer: torch.Tensor,
) -> torch.Tensor:
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

    # Normalize each of bg/surface/fg splat layers separately.
    normalization_scales = 1.0 / (
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        torch.maximum(
            splatted_weights_per_occlusion_layer,
            torch.tensor([1.0], device=device),
        )
    )  # (N, H, W, 1, 3)

    normalized_splatted_colors = (
        splatted_colors_per_occlusion_layer * normalization_scales
    )  # (N, H, W, 4, 3)

    # Use alpha-compositing to compose the splat layers.
    output_colors = torch.cat(
        [background_color, torch.tensor([0.0], device=device)]
    )  # (4), will broadcast to (N, H, W, 4) below.

    for occlusion_layer_id in (-1, -2, -3):
        # Over-compose the bg, surface, and fg occlusion layers. Note that we already
        # multiplied each pixel's RGBA by its own alpha as part of self-splatting in
        # _compute_splatting_colors_and_weights, so we don't re-multiply by alpha here.
        alpha = normalized_splatted_colors[..., 3:4, occlusion_layer_id]  # (N, H, W, 1)
        output_colors = (
            normalized_splatted_colors[..., occlusion_layer_id]
            + (1.0 - alpha) * output_colors
        )
    return output_colors