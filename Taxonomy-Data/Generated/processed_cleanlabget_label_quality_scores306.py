import numpy as np
from scipy.stats import entropy

def get_label_quality_scores(labels, pred_probs, method="self_confidence", adjust_pred_probs=False):
    """
    Compute label quality scores for a multi-class classification dataset.

    Parameters
    ----------
    labels : np.ndarray
        A discrete vector of noisy labels.
    pred_probs : np.ndarray
        An array of shape (N, K) of model-predicted probabilities.
    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
        Label quality scoring method.
    adjust_pred_probs : bool, optional
        Adjust predicted probabilities for class imbalance.

    Returns
    -------
    label_quality_scores : np.ndarray
        Contains one score (between 0 and 1) per example.
    """
    if adjust_pred_probs:
        # Adjust predicted probabilities for class imbalance
        class_counts = np.bincount(labels)
        class_probs = class_counts / len(labels)
        pred_probs = pred_probs / class_probs
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

    scores = np.zeros(len(labels))

    for i, (label, probs) in enumerate(zip(labels, pred_probs)):
        if method == "self_confidence":
            score = probs[label]
        elif method == "normalized_margin":
            max_other_prob = np.max(np.delete(probs, label))
            score = probs[label] - max_other_prob
        elif method == "confidence_weighted_entropy":
            self_confidence = probs[label]
            score = entropy(probs) / self_confidence
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize score to be between 0 and 1
        if method in ["normalized_margin", "self_confidence"]:
            score = max(0, min(1, score))
        elif method == "confidence_weighted_entropy":
            score = 1 - score  # Invert to ensure lower scores indicate likely mislabeled data

        scores[i] = score

    return scores