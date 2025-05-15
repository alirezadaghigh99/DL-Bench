def solarize(img: Tensor, threshold: float) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    _assert_channels(img, [1, 3])

    if threshold > _max_value(img.dtype):
        raise TypeError("Threshold should be less than bound of img.")

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)