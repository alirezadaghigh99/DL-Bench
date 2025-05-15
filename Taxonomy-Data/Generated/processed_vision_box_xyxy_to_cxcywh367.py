import torch

def _box_xyxy_to_cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of boxes,
                              and each box is represented as (x1, y1, x2, y2).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each box is represented as (cx, cy, w, h).
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes should be a torch.Tensor")

    # Calculate the center coordinates (cx, cy)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    # Calculate the width (w) and height (h)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # Stack the results into a new tensor
    converted_boxes = torch.stack((cx, cy, w, h), dim=1)

    return converted_boxes

