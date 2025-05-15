import torch

def _box_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of boxes,
                              and each box is represented by (cx, cy, w, h).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where each box is represented by (x1, y1, x2, y2).
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes should be a torch.Tensor")

    # Extract the center coordinates and dimensions
    cx, cy, w, h = boxes.unbind(-1)

    # Calculate the top-left corner (x1, y1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h

    # Calculate the bottom-right corner (x2, y2)
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    # Stack the results into a single tensor
    return torch.stack((x1, y1, x2, y2), dim=-1)

