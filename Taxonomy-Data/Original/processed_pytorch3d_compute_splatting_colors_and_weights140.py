def _compute_splatting_colors_and_weights(
    pixel_coords_screen: torch.Tensor,
    colors: torch.Tensor,
    sigma: float,
    offsets: torch.Tensor,
) -> torch.Tensor:
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
    N, H, W, K, C = colors.shape
    splat_kernel_normalization = _get_splat_kernel_normalization(offsets, sigma)

    # Distance from each barycentric-interpolated triangle vertices' triplet from its
    # "ideal" pixel-center location. pixel_coords_screen are in screen coordinates, and
    # should be at the "ideal" locations on the forward pass -- e.g.
    # pixel_coords_screen[n, 24, 31, k] = [24.5, 31.5]. For this reason, q_to_px_center
    # should equal torch.zeros during the forward pass. On the backwards pass, these
    # coordinates will be adjusted and non-zero, allowing the gradients to flow back
    # to the mesh vertex coordinates.
    q_to_px_center = (
        torch.floor(pixel_coords_screen[..., :2]) - pixel_coords_screen[..., :2] + 0.5
    ).view((N, H, W, K, 1, 2))

    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    dist2_p_q = torch.sum((q_to_px_center + offsets) ** 2, dim=5)  # (N, H, W, K, 9)
    splat_weights = torch.exp(-dist2_p_q / (2 * sigma**2))
    alpha = colors[..., 3:4]
    splat_weights = (alpha * splat_kernel_normalization * splat_weights).unsqueeze(
        5
    )  # (N, H, W, K, 9, 1)

    # splat_colors[n, h, w, direction, :] contains the splatting color (weighted by the
    # splatting weight) that pixel h, w will splat in one  of the nine possible
    # directions (e.g. nhw0 corresponds to splatting in [-1, 1] direciton, nhw4 is
    # self-splatting).
    splat_colors = splat_weights * colors.unsqueeze(4)  # (N, H, W, K, 9, 4)

    return torch.cat([splat_colors, splat_weights], dim=5)