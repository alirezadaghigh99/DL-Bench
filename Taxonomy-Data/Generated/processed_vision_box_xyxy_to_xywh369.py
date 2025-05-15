import torch

def _box_xyxy_to_xywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of boxes,
                              and each box is represented as (x1, y1, x2, y2).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each box is represented as (x, y, w, h).
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes should be a torch.Tensor")

    # Calculate the width and height
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # Calculate the top-left corner (x, y)
    x = boxes[:, 0]
    y = boxes[:, 1]

    # Stack the results into a new tensor
    boxes_xywh = torch.stack((x, y, w, h), dim=1)

    return boxes_xywh

