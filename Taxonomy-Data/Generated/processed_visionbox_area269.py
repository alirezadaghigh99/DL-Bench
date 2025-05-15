import torch

def box_area(boxes):
    """
    Computes the area of bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) where each row is (x1, y1, x2, y2).

    Returns:
        torch.Tensor: A tensor of shape (N) containing the area of each box.
    """
    # Ensure the input is a tensor
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes must be a torch.Tensor")

    # Check the shape of the input tensor
    if boxes.ndimension() != 2 or boxes.size(1) != 4:
        raise ValueError("Input boxes must have shape (N, 4)")

    # Calculate the width and height of each box
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Calculate the area of each box
    areas = widths * heights

    return areas

