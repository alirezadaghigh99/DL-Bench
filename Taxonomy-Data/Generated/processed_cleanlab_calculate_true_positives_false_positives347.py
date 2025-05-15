import numpy as np

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou

def _calculate_true_positives_false_positives(pred_bboxes, lab_bboxes, iou_threshold=0.5, return_false_negative=False):
    """
    Calculate true positives, false positives, and optionally false negatives for object detection tasks.
    
    Parameters:
    - pred_bboxes: numpy array of predicted bounding boxes (N x 4)
    - lab_bboxes: numpy array of ground truth bounding boxes (M x 4)
    - iou_threshold: float, IoU threshold to consider a detection as true positive
    - return_false_negative: bool, whether to return false negatives
    
    Returns:
    - true_positives: numpy array of true positives
    - false_positives: numpy array of false positives
    - false_negatives: numpy array of false negatives (if return_false_negative is True)
    """
    num_preds = len(pred_bboxes)
    num_labels = len(lab_bboxes)

    true_positives = np.zeros(num_preds, dtype=bool)
    false_positives = np.zeros(num_preds, dtype=bool)
    false_negatives = np.zeros(num_labels, dtype=bool) if return_false_negative else None

    if num_labels == 0:
        # If there are no ground truth boxes, all predictions are false positives
        false_positives[:] = True
        return (true_positives, false_positives, false_negatives) if return_false_negative else (true_positives, false_positives)

    for pred_idx, pred_box in enumerate(pred_bboxes):
        best_iou = 0
        best_label_idx = -1
        for label_idx, lab_box in enumerate(lab_bboxes):
            iou = calculate_iou(pred_box, lab_box)
            if iou > best_iou:
                best_iou = iou
                best_label_idx = label_idx

        if best_iou >= iou_threshold:
            true_positives[pred_idx] = True
            if return_false_negative:
                false_negatives[best_label_idx] = True
        else:
            false_positives[pred_idx] = True

    if return_false_negative:
        false_negatives = ~false_negatives

    return (true_positives, false_positives, false_negatives) if return_false_negative else (true_positives, false_positives)

