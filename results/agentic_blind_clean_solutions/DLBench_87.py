"""Helper methods used internally for computing label quality scores."""
import warnings
import numpy as np
from typing import Optional
from scipy.special import xlogy
from cleanlab.count import get_confident_thresholds

def _subtract_confident_thresholds(labels: Optional[np.ndarray], pred_probs: np.ndarray, multi_label: bool=False, confident_thresholds: Optional[np.ndarray]=None) -> np.ndarray:
    """You need to implement a function named `_subtract_confident_thresholds` that adjusts predicted probabilities by subtracting class-specific confidence thresholds and then re-normalizing the probabilities. This adjustment aims to handle class imbalance in classification tasks. The function accepts labels, predicted probabilities, an optional flag for multi-label settings, and pre-calculated confidence thresholds. If confidence thresholds are not provided, they will be calculated from the labels and predicted probabilities using the `get_confident_thresholds` method. After subtracting the thresholds, the function ensures no negative values by shifting and then re-normalizing the probabilities. The function returns the adjusted predicted probabilities as a NumPy array. If neither labels nor pre-calculated thresholds are provided, a `ValueError` is raised."""
    if confident_thresholds is None:
        if labels is None:
            raise ValueError(
                "Cannot subtract confident thresholds without labels. "
                "Either pass in the labels parameter or provide pre-calculated confident_thresholds."
            )
        confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)

    # For multi-label, confident_thresholds has shape (K, 2); use the positive class thresholds
    if multi_label:
        thresholds = confident_thresholds[:, 1]
    else:
        thresholds = confident_thresholds

    # Subtract class-specific thresholds from predicted probabilities
    pred_probs = pred_probs - thresholds

    # Shift each row up to eliminate any negative values
    pred_probs -= pred_probs.min(axis=1, keepdims=True)

    # Renormalize each row to sum to 1
    row_sums = pred_probs.sum(axis=1, keepdims=True)
    pred_probs /= row_sums

    return pred_probs

def get_normalized_entropy(pred_probs: np.ndarray, min_allowed_prob: Optional[float]=None) -> np.ndarray:
    """Return the normalized entropy of pred_probs.

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.

    Read more about normalized entropy `on Wikipedia <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.

    Parameters
    ----------
    pred_probs : np.ndarray (shape (N, K))
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class: P(label=k|x)

    min_allowed_prob : float, default: None, deprecated
      Minimum allowed probability value. If not `None` (default),
      entries of `pred_probs` below this value will be clipped to this value.

      .. deprecated:: 2.5.0
         This keyword is deprecated and should be left to the default.
         The entropy is well-behaved even if `pred_probs` contains zeros,
         clipping is unnecessary and (slightly) changes the results.

    Returns
    -------
    entropy : np.ndarray (shape (N, ))
      Each element is the normalized entropy of the corresponding row of ``pred_probs``.

    Raises
    ------
    ValueError
        An error is raised if any of the probabilities is not in the interval [0, 1].
    """
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError('All probabilities are required to be in the interval [0, 1].')
    num_classes = pred_probs.shape[1]
    if min_allowed_prob is not None:
        warnings.warn('Using `min_allowed_prob` is not necessary anymore and will be removed.', DeprecationWarning)
        pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)
