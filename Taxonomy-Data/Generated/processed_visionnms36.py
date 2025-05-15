import torch
import torchvision

def nms(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression (NMS) on bounding boxes.

    Args:
        boxes (Tensor[N, 4]): Bounding box coordinates in (x1, y1, x2, y2) format.
        scores (Tensor[N]): Scores for each box.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        Tensor: Indices of the boxes that have been kept after NMS, sorted in decreasing order of scores.
    """
    # Check if torchvision is available and has the nms function
    if not hasattr(torchvision.ops, 'nms'):
        raise ImportError("torchvision.ops.nms is not available. Please ensure you have torchvision installed.")

    # Log API usage
    print("Using torchvision.ops.nms for non-maximum suppression.")

    # Perform non-maximum suppression using torchvision's built-in function
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

    return keep_indices

