def elastic_transform(
    img: Tensor,
    displacement: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Transform a tensor image with elastic transformations.
    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to grid_sample from the image.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        img (PIL Image or Tensor): Image on which elastic_transform is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".
        displacement (Tensor): The displacement field. Expected shape is [1, H, W, 2].
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(elastic_transform)
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(displacement, torch.Tensor):
        raise TypeError("Argument displacement should be a Tensor")

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not F_pil._is_pil_image(img):
            raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
        t_img = pil_to_tensor(img)

    shape = t_img.shape
    shape = (1,) + shape[-2:] + (2,)
    if shape != displacement.shape:
        raise ValueError(f"Argument displacement shape should be {shape}, but given {displacement.shape}")

    # TODO: if image shape is [N1, N2, ..., C, H, W] and
    # displacement is [1, H, W, 2] we need to reshape input image
    # such grid_sampler takes internal code for 4D input

    output = F_t.elastic_transform(
        t_img,
        displacement,
        interpolation=interpolation.value,
        fill=fill,
    )

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output, mode=img.mode)
    return output