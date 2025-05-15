def _load_image_to_tensor(path_file: Path, device: Device) -> Tensor:
    """Read an image file and decode using the Kornia Rust backend.

    The decoded image is returned as numpy array with shape HxWxC.

    Args:
        path_file: Path to a valid image file.
        device: the device where you want to get your image placed.

    Return:
        Image tensor with shape :math:`(3,H,W)`.
    """

    # read image and return as `np.ndarray` with shape HxWxC
    if path_file.suffix.lower() in [".jpg", ".jpeg"]:
        img = kornia_rs.read_image_jpeg(str(path_file))
    else:
        img = kornia_rs.read_image_any(str(path_file))

    # convert the image to tensor with shape CxHxW
    img_t = image_to_tensor(img, keepdim=True)

    # move the tensor to the desired device,
    dev = device if isinstance(device, torch.device) or device is None else torch.device(device)

    return img_t.to(device=dev)