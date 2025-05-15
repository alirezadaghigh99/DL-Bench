def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The image values should be uint8 in [0, 255] or float in [0, 1].
    Keypoints can be drawn for multiple instances at a time.

    This method allows that keypoints and their connectivity are drawn based on the visibility of this keypoint.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoint locations for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where each tuple contains a pair of keypoints
            to be connected.
            If at least one of the two connected keypoints has a ``visibility`` of False,
            this specific connection is not drawn.
            Exclusions due to invisibility are computed per-instance.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
        visibility (Tensor): Tensor of shape (num_instances, K) specifying the visibility of the K
            keypoints for each of the N instances.
            True means that the respective keypoint is visible and should be drawn.
            False means invisible, so neither the point nor possible connections containing it are drawn.
            The input tensor will be cast to bool.
            Default ``None`` means that all the keypoints are visible.
            For more details, see :ref:`draw_keypoints_with_visibility`.

    Returns:
        img (Tensor[C, H, W]): Image Tensor with keypoints drawn.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_keypoints)
    # validate image
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(f"The image dtype must be uint8 or float, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # validate keypoints
    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    # validate visibility
    if visibility is None:  # set default
        visibility = torch.ones(keypoints.shape[:-1], dtype=torch.bool)
    if visibility.ndim == 3:
        # If visibility was passed as pred.split([2, 1], dim=-1), it will be of shape (num_instances, K, 1).
        # We make sure it is of shape (num_instances, K). This isn't documented, we're just being nice.
        visibility = visibility.squeeze(-1)
    if visibility.ndim != 2:
        raise ValueError(f"visibility must be of shape (num_instances, K). Got ndim={visibility.ndim}")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError(
            "keypoints and visibility must have the same dimensionality for num_instances and K. "
            f"Got {visibility.shape = } and {keypoints.shape = }"
        )

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        from torchvision.transforms.v2.functional import to_dtype  # noqa

        image = to_dtype(image, dtype=torch.uint8, scale=True)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()
    img_vis = visibility.cpu().bool().tolist()

    for kpt_inst, vis_inst in zip(img_kpts, img_vis):
        for kpt_coord, kp_vis in zip(kpt_inst, vis_inst):
            if not kp_vis:
                continue
            x1 = kpt_coord[0] - radius
            x2 = kpt_coord[0] + radius
            y1 = kpt_coord[1] - radius
            y2 = kpt_coord[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                if (not vis_inst[connection[0]]) or (not vis_inst[connection[1]]):
                    continue
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    out = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
    if original_dtype.is_floating_point:
        out = to_dtype(out, dtype=original_dtype, scale=True)
    return out