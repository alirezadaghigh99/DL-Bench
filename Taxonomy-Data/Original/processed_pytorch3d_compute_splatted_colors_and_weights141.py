def _compute_splatted_colors_and_weights(
    occlusion_layers: torch.Tensor,  # (N, H, W, 9)
    splat_colors_and_weights: torch.Tensor,  # (N, H, W, K, 9, 5)
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    N, H, W, K, _, _ = splat_colors_and_weights.shape

    # Create an occlusion mask, with the last dimension of length 3, corresponding to
    # background/surface/foreground splatting. E.g. occlusion_layer_mask[n,h,w,k,d,0] is
    # 1 if the pixel at hw is splatted from direction d such that the splatting pixel p
    # is below the splatted pixel q (in the background); otherwise, the value is 0.
    # occlusion_layer_mask[n,h,w,k,d,1] is 1 if the splatting pixel is at the same
    # surface level as the splatted pixel q, and occlusion_layer_mask[n,h,w,k,d,2] is
    # 1 only if the splatting pixel is in the foreground.
    layer_ids = torch.arange(K, device=splat_colors_and_weights.device).view(
        1, 1, 1, K, 1
    )
    occlusion_layers = occlusion_layers.view(N, H, W, 1, 9)
    occlusion_layer_mask = torch.stack(
        [
            occlusion_layers > layer_ids,  # (N, H, W, K, 9)
            occlusion_layers == layer_ids,  # (N, H, W, K, 9)
            occlusion_layers < layer_ids,  # (N, H, W, K, 9)
        ],
        dim=5,
    ).float()  # (N, H, W, K, 9, 3)

    # (N * H * W, 5, 9 * K) x (N * H * W, 9 * K, 3) -> (N * H * W, 5, 3)
    splatted_colors_and_weights = torch.bmm(
        splat_colors_and_weights.permute(0, 1, 2, 5, 3, 4).reshape(
            (N * H * W, 5, K * 9)
        ),
        occlusion_layer_mask.reshape((N * H * W, K * 9, 3)),
    ).reshape((N, H, W, 5, 3))

    return (
        splatted_colors_and_weights[..., :4, :],
        splatted_colors_and_weights[..., 4:5, :],
    )