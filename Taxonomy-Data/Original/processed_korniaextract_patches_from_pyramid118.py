def extract_patches_from_pyramid(
    img: Tensor, laf: Tensor, PS: int = 32, normalize_lafs_before_extraction: bool = True
) -> Tensor:
    """Extract patches defined by LAFs from image tensor.

    Patches are extracted from appropriate pyramid level.

    Args:
        img: images, LAFs are detected in  :math:`(B, CH, H, W)`.
        laf: :math:`(B, N, 2, 3)`.
        PS: patch size.
        normalize_lafs_before_extraction: if True, lafs are normalized to image size.

    Returns:
        patches with shape :math:`(B, N, CH, PS,PS)`.
    """
    KORNIA_CHECK_LAF(laf)
    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    _, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    max_level = min(img.size(2), img.size(3)) // PS
    pyr_idx = scale.log2().clamp(min=0.0, max=max(0, max_level - 1)).long()
    cur_img = img
    cur_pyr_level = 0
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    we_are_in_business = True
    while we_are_in_business:
        _, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum().item()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_patch_grid_from_normalized_LAF(cur_img[i : i + 1], nlaf[i : i + 1, scale_mask, :, :], PS)
            patches = F.grid_sample(
                cur_img[i : i + 1].expand(grid.shape[0], ch, h, w), grid, padding_mode="border", align_corners=False
            )
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches.to(nlaf.dtype))
        we_are_in_business = min(cur_img.size(2), cur_img.size(3)) >= PS
        if not we_are_in_business:
            break
        cur_img = pyrdown(cur_img)
        cur_pyr_level += 1
    return out