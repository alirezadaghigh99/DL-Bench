"""
Methods to rank examples in standard (multi-class) classification datasets by cleanlab's `label quality score`.
Except for `~cleanlab.rank.order_label_issues`, which operates only on the subset of the data identified
as potential label issues/errors, the methods in this module can be used on whichever subset
of the dataset you choose (including the entire dataset) and provide a `label quality score` for
every example. You can then do something like: ``np.argsort(label_quality_score)`` to obtain ranked
indices of individual datapoints based on their quality.

Note: multi-label classification is not supported by most methods in this module,
each example must be labeled as belonging to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.
For multi-label classification, instead see :py:func:`multilabel_classification.get_label_quality_scores <cleanlab.multilabel_classification.get_label_quality_scores>`.

Note: Label quality scores are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`. This is encouraged to get better results.
"""
import numpy as np
from sklearn.metrics import log_loss
from typing import List, Optional
import warnings
from cleanlab.internal.validation import assert_valid_inputs
from cleanlab.internal.constants import CLIPPING_LOWER_BOUND
from cleanlab.internal.label_quality_utils import _subtract_confident_thresholds, get_normalized_entropy

def get_label_quality_scores(labels: np.ndarray, pred_probs: np.ndarray, *, method: str='self_confidence', adjust_pred_probs: bool=False) -> np.ndarray:
    """Returns a label quality score for each datapoint.

    This is a function to compute label quality scores for standard (multi-class) classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.
      Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. format: ``labels = np.ndarray([1,0,2,1,1,0...])``.

    pred_probs : np.ndarray, optional
      An array of shape ``(N, K)`` of model-predicted probabilities,
      ``P(label=k|x)``. Each row of this matrix corresponds
      to an example `x` and contains the model-predicted probabilities that
      `x` belongs to each possible class, for each of the K classes. The
      columns must be ordered such that these probabilities correspond to
      class 0, 1, ..., K-1.

      **Note**: Returned label issues are most accurate when they are computed based on out-of-sample `pred_probs` from your model.
      To obtain out-of-sample predicted probabilities for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method.

      Letting ``k = labels[i]`` and ``P = pred_probs[i]`` denote the given label and predicted class-probabilities
      for datapoint *i*, its score can either be:

      - ``'normalized_margin'``: ``P[k] - max_{k' != k}[ P[k'] ]``
      - ``'self_confidence'``: ``P[k]``
      - ``'confidence_weighted_entropy'``: ``entropy(P) / self_confidence``

      Note: the actual label quality scores returned by this method
      may be transformed versions of the above, in order to ensure
      their values lie between 0-1 with lower values indicating more likely mislabeled data.

      Let ``C = {0, 1, ..., K-1}`` be the set of classes specified for our classification task.

      The `normalized_margin` score works better for identifying class conditional label errors,
      i.e. examples for which another label in ``C`` is appropriate but the given label is not.

      The `self_confidence` score works better for identifying alternative label issues
      corresponding to bad examples that are: not from any of the classes in ``C``,
      well-described by 2 or more labels in ``C``,
      or generally just out-of-distribution (i.e. anomalous outliers).

    adjust_pred_probs : bool, optional
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.

    See Also
    --------
    get_self_confidence_for_each_label
    get_normalized_margin_for_each_label
    get_confidence_weighted_entropy_for_each_label
    """
    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=True)
    return _compute_label_quality_scores(labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs)

def _compute_label_quality_scores(labels: np.ndarray, pred_probs: np.ndarray, *, method: str='self_confidence', adjust_pred_probs: bool=False, confident_thresholds: Optional[np.ndarray]=None) -> np.ndarray:
    """Internal implementation of get_label_quality_scores that assumes inputs
    have already been checked and are valid. This speeds things up.
    Can also take in pre-computed confident_thresholds to further accelerate things.
    """
    scoring_funcs = {'self_confidence': get_self_confidence_for_each_label, 'normalized_margin': get_normalized_margin_for_each_label, 'confidence_weighted_entropy': get_confidence_weighted_entropy_for_each_label}
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
        raise ValueError(f'\n            {method} is not a valid scoring method for rank_by!\n            Please choose a valid rank_by: self_confidence, normalized_margin, confidence_weighted_entropy\n            ')
    if adjust_pred_probs:
        if method == 'confidence_weighted_entropy':
            raise ValueError(f'adjust_pred_probs is not currently supported for {method}.')
        pred_probs = _subtract_confident_thresholds(labels=labels, pred_probs=pred_probs, confident_thresholds=confident_thresholds)
    scoring_inputs = {'labels': labels, 'pred_probs': pred_probs}
    label_quality_scores = scoring_func(**scoring_inputs)
    return label_quality_scores

def get_label_quality_ensemble_scores(labels: np.ndarray, pred_probs_list: List[np.ndarray], *, method: str='self_confidence', adjust_pred_probs: bool=False, weight_ensemble_members_by: str='accuracy', custom_weights: Optional[np.ndarray]=None, log_loss_search_T_values: List[float]=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0], verbose: bool=True) -> np.ndarray:
    """Returns a label quality score for each datapoint, based on predictions from an ensemble of models.

    This is a function to compute label quality scores for standard (multi-class) classification datasets,
    where lower scores indicate labels less likely to be correct.

    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
      A discrete vector of noisy labels.
      *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.

    pred_probs_list : List[np.ndarray]
      Each element is an array of shape ``(N, K)`` of model-predicted probabilities from one ensemble member.

    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. See `~cleanlab.rank.get_label_quality_scores`.

    adjust_pred_probs : bool, optional
      Account for class imbalance via adjustment of predicted probabilities.

    weight_ensemble_members_by : {"equal", "accuracy", "log_loss_search", "custom"}, default="accuracy"
      Weighting scheme used to aggregate label quality scores from each model.

      - ``'equal'``: each ensemble member is weighted equally.
      - ``'accuracy'``: each ensemble member is weighted by its accuracy.
      - ``'log_loss_search'``: weights are determined by a temperature-scaled softmax of negative log
        losses, with the temperature chosen via search over ``log_loss_search_T_values``.
      - ``'custom'``: use the weights provided in ``custom_weights``.

    custom_weights : np.ndarray, optional
      An array of shape ``(P,)`` where P is the number of models, providing custom weights.
      Only used when ``weight_ensemble_members_by='custom'``.

    log_loss_search_T_values : List[float], optional
      Temperature values to search over when ``weight_ensemble_members_by='log_loss_search'``.
      For each T, ensemble weights are proportional to ``exp(-T * log_loss_i)``.
      The T that minimizes the combined log loss is selected.

    verbose : bool, optional
      If ``True``, prints the ensemble member weights used.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    for pred_probs in pred_probs_list:
        assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=True)

    label_quality_scores_list = [
        _compute_label_quality_scores(labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs)
        for pred_probs in pred_probs_list
    ]

    if weight_ensemble_members_by == 'equal':
        weights = np.ones(len(pred_probs_list))

    elif weight_ensemble_members_by == 'accuracy':
        weights = np.array([
            np.mean(np.argmax(pred_probs, axis=1) == labels)
            for pred_probs in pred_probs_list
        ])

    elif weight_ensemble_members_by == 'log_loss_search':
        log_losses = np.array([log_loss(labels, pred_probs) for pred_probs in pred_probs_list])
        best_T = log_loss_search_T_values[0]
        best_combined_log_loss = float('inf')
        for T in log_loss_search_T_values:
            # subtract min for numerical stability; does not change normalized weights
            raw_weights = np.exp(-T * (log_losses - log_losses.min()))
            curr_weights = raw_weights / raw_weights.sum()
            combined_pred_probs = np.sum(
                [w * pp for w, pp in zip(curr_weights, pred_probs_list)], axis=0
            )
            curr_log_loss = log_loss(labels, combined_pred_probs)
            if curr_log_loss < best_combined_log_loss:
                best_combined_log_loss = curr_log_loss
                best_T = T
        if verbose:
            print(f'Best temperature T={best_T} found for log_loss_search ensemble weighting.')
        weights = np.exp(-best_T * (log_losses - log_losses.min()))

    elif weight_ensemble_members_by == 'custom':
        if custom_weights is None:
            raise ValueError(
                "custom_weights must be provided when weight_ensemble_members_by='custom'."
            )
        weights = np.array(custom_weights, dtype=float)

    else:
        raise ValueError(
            f"weight_ensemble_members_by must be one of 'equal', 'accuracy', 'log_loss_search', "
            f"or 'custom', but '{weight_ensemble_members_by}' was provided."
        )

    total_weight = weights.sum()
    if total_weight == 0:
        weights = np.ones(len(pred_probs_list)) / len(pred_probs_list)
    else:
        weights = weights / total_weight

    if verbose:
        print(f'Ensemble member weights: {weights}')

    label_quality_scores = np.average(
        np.stack(label_quality_scores_list, axis=0),
        axis=0,
        weights=weights,
    )

    return label_quality_scores

def find_top_issues(quality_scores: np.ndarray, *, top: int=10) -> np.ndarray:
    """Returns the sorted indices of the `top` issues in `quality_scores`, ordered from smallest to largest quality score
    (i.e., from most to least likely to be an issue). For example, the first value returned is the index corresponding
    to the smallest value in `quality_scores` (most likely to be an issue). The second value in the returned array is
    the index corresponding to the second smallest value in `quality-scores` (second-most likely to be an issue), and so forth.

    This method assumes that `quality_scores` shares an index with some dataset such that the indices returned by this method
    map to the examples in that dataset.

    Parameters
    ----------
    quality_scores :
      Array of shape ``(N,)``, where N is the number of examples, containing one quality score for each example in the dataset.

    top :
      The number of indices to return.

    Returns
    -------
    top_issue_indices :
      Indices of top examples most likely to suffer from an issue (ranked by issue severity)."""
    if top is None or top > len(quality_scores):
        top = len(quality_scores)
    top_outlier_indices = quality_scores.argsort()[:top]
    return top_outlier_indices

def order_label_issues(label_issues_mask: np.ndarray, labels: np.ndarray, pred_probs: np.ndarray, *, rank_by: str='self_confidence', rank_by_kwargs: dict={}) -> np.ndarray:
    """Sorts label issues by label quality score.

    Default label quality score is "self_confidence".

    Parameters
    ----------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a label
      issue and ``False`` represents an example that is accurately labeled with
      high confidence.

    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray (shape (N, K))
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    rank_by : str, optional
      Score by which to order label error indices (in increasing order). See
      the `method` argument of `~cleanlab.rank.get_label_quality_scores`.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into `~cleanlab.rank.get_label_quality_scores` function.
      Accepted args include `adjust_pred_probs`.

    Returns
    -------
    label_issues_idx : np.ndarray
      Return an array of the indices of the examples with label issues,
      ordered by the label-quality scoring method passed to `rank_by`.
    """
    allow_one_class = False
    if isinstance(labels, np.ndarray) or all((isinstance(lab, int) for lab in labels)):
        if set(labels) == {0}:
            allow_one_class = True
    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=allow_one_class)
    label_issues_idx = np.arange(len(labels))[label_issues_mask]
    label_quality_scores = get_label_quality_scores(labels, pred_probs, method=rank_by, **rank_by_kwargs)
    label_quality_scores_issues = label_quality_scores[label_issues_mask]
    return label_issues_idx[np.argsort(label_quality_scores_issues)]

def get_self_confidence_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """Returns the self-confidence label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    The self-confidence is the classifier's predicted probability that an example belongs to
    its given class label.

    Self-confidence can work better than normalized-margin for detecting label errors due to out-of-distribution (OOD) or weird examples
    vs. label errors in which labels for random examples have been replaced by other classes.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    return pred_probs[np.arange(labels.shape[0]), labels]

def get_normalized_margin_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """Returns the "normalized margin" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Letting ``k`` denote the given label for a datapoint, the margin is
    ``(p(label = k) - max(p(label != k)))``, i.e. the probability
    of the given label minus the probability of the argmax label that is not
    the given label (``margin = prob_label - max_prob_not_label``).
    This gives you an idea of how likely an example is BOTH its given label AND not another label,
    and therefore, scores its likelihood of being a good label or a label error.
    The normalized margin is simply a transformed version of the margin,
    to ensure values between 0-1 with lower values indicating more likely mislabeled data.

    Normalized margin works best for finding class conditional label errors where
    there is another label in the set of classes that is clearly better than the given label.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    (N, K) = pred_probs.shape
    del_indices = np.arange(N) * K + labels
    max_prob_not_label = np.max(np.delete(pred_probs, del_indices, axis=None).reshape(N, K - 1), axis=-1)
    label_quality_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_quality_scores

def get_confidence_weighted_entropy_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """Returns the "confidence weighted entropy" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    "confidence weighted entropy" is defined as the normalized entropy divided by "self-confidence".
    The returned values are a transformed version of this score, in order to
    ensure values between 0-1 with lower values indicating more likely mislabeled data.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    self_confidence = np.clip(self_confidence, a_min=CLIPPING_LOWER_BOUND, a_max=None)
    label_quality_scores = get_normalized_entropy(pred_probs) / self_confidence
    clipped_scores = np.clip(label_quality_scores, a_min=CLIPPING_LOWER_BOUND, a_max=None)
    label_quality_scores = np.log(label_quality_scores + 1) / clipped_scores
    return label_quality_scores
