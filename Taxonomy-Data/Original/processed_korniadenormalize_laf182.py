def denormalize_laf(LAF: Tensor, images: Tensor) -> Tensor:
    """De-normalize LAFs from scale to image scale. The convention is that center of 5-pixel image (coordinates
    from 0 to 4) is 2, and not 2.5.

        B,N,H,W = images.size()
        MIN_SIZE = min(H - 1, W -1)
        [a11 a21 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a21*MIN_SIZE x*(W-1)]
        [a21*MIN_SIZE a22*MIN_SIZE y*(W-1)]

    Args:
        LAF: :math:`(B, N, 2, 3)`
        images: :math:`(B, CH, H, W)`

    Returns:
        the denormalized LAF: :math:`(B, N, 2, 3)`, scale in pixels
    """
    KORNIA_CHECK_LAF(LAF)
    _, _, h, w = images.size()
    wf = float(w - 1)
    hf = float(h - 1)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3, dtype=LAF.dtype, device=LAF.device) * min_size
    coef[0, 0, 0, 2] = wf
    coef[0, 0, 1, 2] = hf
    return coef.expand_as(LAF) * LAF