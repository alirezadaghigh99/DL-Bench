def _compute_occlusion_layers(
    q_depth: torch.Tensor,
) -> torch.Tensor:
    """
    For each splatting pixel, decide whether it splats from a background, surface, or
    foreground depth relative to the splatted pixel. See unit tests in
    test_splatter_blend for some enlightening examples.

    Args:
        q_depth: (N, H, W, K) tensor of z-values of the splatted pixels.

    Returns:
        occlusion_layers: (N, H, W, 9) long tensor. Each of the 9 values corresponds to
            one of the nine splatting directions ([-1, -1], [-1, 0], ..., [1,
            1]). The value at nhwd (where d is the splatting direction) is 0 if
            the splat in direction d is on the same surface level as the pixel at
            hw. The value is negative if the splat is in the background (occluded
            by another splat above it that is at the same surface level as the
            pixel splatted on), and the value is positive if the splat is in the
            foreground.
    """
    N, H, W, K = q_depth.shape

    # q are the "center pixels" and p the pixels splatting onto them. Use `unfold` to
    # create `p_depth`, a tensor with 9 layers, each of which corresponds to the
    # depth of a neighbor of q in one of the 9 directions. For example, p_depth[nk0hw]
    # is the depth of the pixel splatting onto pixel nhwk from the [-1, -1] direction,
    # and p_depth[nk4hw] the depth of q (self-splatting onto itself).
    # More concretely, imagine the pixel depths in a 2x2 image's k-th layer are
    #   .1 .2
    #   .3 .4
    # Then (remembering that we pad with zeros when a pixel has fewer than 9 neighbors):
    #
    # p_depth[n, k, :, 0, 0] = [ 0  0  0  0 .1 .2  0 .3 .4] - neighbors of .1
    # p_depth[n, k, :, 0, 1] = [ 0  0  0 .1 .2  0 .3 .4  0] - neighbors of .2
    # p_depth[n, k, :, 1, 0] = [ 0 .1 .2  0 .3 .4  0  0  0] - neighbors of .3
    # p_depth[n, k, :, 0, 1] = [.1 .2  0 .3 .4  0  0  0  0] - neighbors of .4
    q_depth = q_depth.permute(0, 3, 1, 2)  # (N, K, H, W)
    p_depth = F.unfold(q_depth, kernel_size=3, padding=1)  # (N, 3^2 * K, H * W)
    q_depth = q_depth.view(N, K, 1, H, W)
    p_depth = p_depth.view(N, K, 9, H, W)

    # Take the center pixel q's top rasterization layer. This is the "surface layer"
    # that we're splatting on. For each of the nine splatting directions p, find which
    # of the K splatting rasterization layers is closest in depth to the surface
    # splatted layer.
    qtop_to_p_zdist = torch.abs(p_depth - q_depth[:, 0:1])  # (N, K, 9, H, W)
    qtop_to_p_closest_zdist, qtop_to_p_closest_id = qtop_to_p_zdist.min(dim=1)

    # For each of the nine splatting directions p, take the top of the K rasterization
    # layers. Check which of the K q-layers (that the given direction is splatting on)
    # is closest in depth to the top splatting layer.
    ptop_to_q_zdist = torch.abs(p_depth[:, 0:1] - q_depth)  # (N, K, 9, H, W)
    ptop_to_q_closest_zdist, ptop_to_q_closest_id = ptop_to_q_zdist.min(dim=1)

    # Decide whether each p is on the same level, below, or above the q it is splatting
    # on. See Fig. 4 in [0] for an illustration. Briefly: say we're interested in pixel
    # p_{h, w} = [10, 32] splatting onto its neighbor q_{h, w} = [11, 33]. The splat is
    # coming from direction [-1, -1], which has index 0 in our enumeration of splatting
    # directions. Hence, we are interested in
    #
    # P = p_depth[n, :, d=0, 11, 33] - a vector of K depth values, and
    # Q = q_depth.squeeze()[n, :, 11, 33] - a vector of K depth values.
    #
    # If Q[0] is closest, say, to P[2], then we assume the 0th surface layer of Q is
    # the same surface as P[2] that's splatting onto it, and P[:2] are foreground splats
    # and P[3:] are background splats.
    #
    # If instead say Q[2] is closest to P[0], then all the splats are background splats,
    # because the top splatting layer is the same surface as a non-top splatted layer.
    #
    # Finally, if Q[0] is closest to P[0], then the top-level P is splatting onto top-
    # level Q, and P[1:] are all background splats.
    occlusion_offsets = torch.where(  # noqa
        ptop_to_q_closest_zdist < qtop_to_p_closest_zdist,
        -ptop_to_q_closest_id,
        qtop_to_p_closest_id,
    )  # (N, 9, H, W)

    occlusion_layers = occlusion_offsets.permute((0, 2, 3, 1))  # (N, H, W, 9)
    return occlusion_layers