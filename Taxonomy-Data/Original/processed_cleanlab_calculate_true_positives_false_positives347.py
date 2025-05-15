def _calculate_true_positives_false_positives(
    pred_bboxes: np.ndarray,
    lab_bboxes: np.ndarray,
    iou_threshold: Optional[float] = 0.5,
    return_false_negative: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates true positives (TP) and false positives (FP) for object detection tasks.
    It takes predicted bounding boxes, ground truth bounding boxes, and an optional Intersection over Union (IoU) threshold as inputs.
    If return_false_negative is True, it returns an array of False negatives as well.
    """
    num_preds = pred_bboxes.shape[0]
    num_labels = lab_bboxes.shape[0]
    num_scales = 1
    true_positives = np.zeros((num_scales, num_preds), dtype=np.float32)
    false_positives = np.zeros((num_scales, num_preds), dtype=np.float32)

    if lab_bboxes.shape[0] == 0:
        false_positives[...] = 1
        if return_false_negative:
            return true_positives, false_positives, np.array([], dtype=np.float32)
        else:
            return true_positives, false_positives
    ious = _get_overlap_matrix(pred_bboxes, lab_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sorted_indices = np.argsort(-pred_bboxes[:, -1])
    is_covered = np.zeros(num_labels, dtype=bool)
    for index in sorted_indices:
        if ious_max[index] >= iou_threshold:
            matching_label = ious_argmax[index]
            if not is_covered[matching_label]:
                is_covered[matching_label] = True
                true_positives[0, index] = 1
            else:
                false_positives[0, index] = 1
        else:
            false_positives[0, index] = 1
    if return_false_negative:
        false_negatives = np.zeros((num_scales, num_labels), dtype=np.float32)
        for label_index in range(num_labels):
            if not is_covered[label_index]:
                false_negatives[0, label_index] = 1
        return true_positives, false_positives, false_negatives
    return true_positives, false_positives