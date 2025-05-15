import torch

def _box_xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where N is the number of boxes,
                              and each box is represented as (x, y, w, h).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) with boxes in (x1, y1, x2, y2) format.
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes should be a torch.Tensor")

    # Check if the input tensor has the correct shape
    if boxes.ndimension() != 2 or boxes.size(1) != 4:
        raise ValueError("Input boxes should have shape (N, 4)")

    # Extract x, y, w, h
    x, y, w, h = boxes.unbind(dim=1)

    # Calculate x1, y1, x2, y2
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # Stack the results into a new tensor
    return torch.stack((x1, y1, x2, y2), dim=1)

