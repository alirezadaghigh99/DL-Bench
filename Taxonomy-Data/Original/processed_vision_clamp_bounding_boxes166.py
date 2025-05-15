def _clamp_bounding_boxes(
    bounding_boxes: torch.Tensor, format: BoundingBoxFormat, canvas_size: Tuple[int, int]
) -> torch.Tensor:
    # TODO: Investigate if it makes sense from a performance perspective to have an implementation for every
    #  BoundingBoxFormat instead of converting back and forth
    in_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    xyxy_boxes = convert_bounding_box_format(
        bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True
    )
    xyxy_boxes[..., 0::2].clamp_(min=0, max=canvas_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=canvas_size[0])
    out_boxes = convert_bounding_box_format(
        xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True
    )
    return out_boxes.to(in_dtype)