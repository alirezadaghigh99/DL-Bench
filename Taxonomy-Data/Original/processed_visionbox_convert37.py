def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Converts :class:`torch.Tensor` boxes from a given ``in_fmt`` to ``out_fmt``.

    .. note::
        For converting a :class:`torch.Tensor` or a :class:`~torchvision.tv_tensors.BoundingBoxes` object
        between different formats,
        consider using :func:`~torchvision.transforms.v2.functional.convert_bounding_box_format` instead.
        Or see the corresponding transform :func:`~torchvision.transforms.v2.ConvertBoundingBoxFormat`.

    Supported ``in_fmt`` and ``out_fmt`` strings are:

    ``'xyxy'``: boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    ``'xywh'``: boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    ``'cxcywh'``: boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(box_convert)
    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError("Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

    if in_fmt == out_fmt:
        return boxes.clone()

    if in_fmt != "xyxy" and out_fmt != "xyxy":
        # convert to xyxy and change in_fmt xyxy
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
        in_fmt = "xyxy"

    if in_fmt == "xyxy":
        if out_fmt == "xywh":
            boxes = _box_xyxy_to_xywh(boxes)
        elif out_fmt == "cxcywh":
            boxes = _box_xyxy_to_cxcywh(boxes)
    elif out_fmt == "xyxy":
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
    return boxes