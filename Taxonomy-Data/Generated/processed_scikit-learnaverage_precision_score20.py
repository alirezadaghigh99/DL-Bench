import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target

def _binary_uninterpolated_average_precision(y_true, y_score, sample_weight=None):
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]
    else:
        sample_weight = np.ones_like(y_true)

    # Calculate precision and recall
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
    fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]

    precisions = tps / (tps + fps)
    recalls = tps / tps[-1]

    # Calculate average precision
    average_precision = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return average_precision

def average_precision_score(y_true, y_score, average='macro', pos_label=1, sample_weight=None):
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score, sample_weight)

    y_type = type_of_target(y_true)
    if y_type == "binary":
        return _binary_uninterpolated_average_precision(y_true, y_score, sample_weight)

    elif y_type == "multiclass" or y_type == "multilabel-indicator":
        if y_type == "multiclass":
            classes = np.unique(y_true)
            y_true = label_binarize(y_true, classes=classes)
        else:
            classes = np.arange(y_true.shape[1])

        ap_scores = []
        for i, class_label in enumerate(classes):
            ap = _binary_uninterpolated_average_precision(
                y_true[:, i], y_score[:, i], sample_weight
            )
            ap_scores.append(ap)

        if average == 'macro':
            return np.mean(ap_scores)
        elif average == 'weighted':
            weights = np.sum(y_true, axis=0)
            return np.average(ap_scores, weights=weights)
        elif average == 'micro':
            y_true = y_true.ravel()
            y_score = y_score.ravel()
            return _binary_uninterpolated_average_precision(y_true, y_score, sample_weight)
        elif average == 'samples':
            return np.mean([_binary_uninterpolated_average_precision(y_true[i], y_score[i], sample_weight) for i in range(y_true.shape[0])])
        else:
            raise ValueError("average has to be one of ['micro', 'samples', 'weighted', 'macro', None]")

    else:
        raise ValueError(f"Unsupported target type: {y_type}")

