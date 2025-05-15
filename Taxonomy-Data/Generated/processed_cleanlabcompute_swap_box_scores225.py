import numpy as np

def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate the intersection coordinates
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    # Calculate the area of intersection
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def compute_swap_box_scores(labels, predictions, alpha=0.5, high_probability_threshold=0.8, 
                            overlapping_label_check=True, auxiliary_inputs=None):
    swap_scores = []

    for i, (label_dict, prediction_array) in enumerate(zip(labels, predictions)):
        image_scores = []
        annotated_boxes = label_dict.get('boxes', [])
        predicted_boxes = prediction_array[:, :4]
        predicted_probs = prediction_array[:, 4:]

        for j, annotated_box in enumerate(annotated_boxes):
            max_iou = 0
            best_pred_idx = -1
            for k, predicted_box in enumerate(predicted_boxes):
                iou = compute_iou(annotated_box, predicted_box)
                if iou > max_iou:
                    max_iou = iou
                    best_pred_idx = k

            if best_pred_idx == -1:
                # No matching prediction found
                score = 0.0
            else:
                # Calculate the score based on IoU and prediction confidence
                predicted_prob = np.max(predicted_probs[best_pred_idx])
                score = alpha * max_iou + (1 - alpha) * predicted_prob

                # Adjust score based on high probability threshold
                if predicted_prob < high_probability_threshold:
                    score *= 0.5

                # Check for overlapping label consistency if required
                if overlapping_label_check:
                    for k, other_annotated_box in enumerate(annotated_boxes):
                        if k != j:
                            iou_with_other = compute_iou(annotated_box, other_annotated_box)
                            if iou_with_other > 0.5:  # Arbitrary overlap threshold
                                score *= 0.5

            # Incorporate auxiliary inputs if provided
            if auxiliary_inputs:
                aux_info = auxiliary_inputs[i].get('info', 1.0)
                score *= aux_info

            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            image_scores.append(score)

        swap_scores.append(np.array(image_scores))

    return swap_scores