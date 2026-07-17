"""
Methods to score the quality of each label in a regression dataset. These can be used to rank the examples whose Y-value is most likely erroneous.

Note: Label quality scores are most accurate when they are computed based on out-of-sample `predictions` from your regression model.
To obtain out-of-sample predictions for every datapoint in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`. This is encouraged to get better results.

If you have a sklearn-compatible regression model, consider using `cleanlab.regression.learn.CleanLearning` instead, which can more accurately identify noisy label values.
"""
from typing import Dict, Callable, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from cleanlab.internal.neighbor.metric import decide_euclidean_metric
from cleanlab.internal.neighbor.knn_graph import features_to_knn
from cleanlab.outlier import OutOfDistribution
from cleanlab.internal.regression_utils import assert_valid_prediction_inputs
from cleanlab.internal.constants import TINY_VALUE

def get_label_quality_scores(labels: ArrayLike, predictions: ArrayLike, *, method: str='outre') -> np.ndarray:
    """Create a Python function called get_label_quality_scores that calculates label quality scores for each example in a regression dataset. The function takes in two array-like inputs: labels (raw labels from the original dataset) and predictions (predicted labels for each example). Additionally, the function has an optional keyword argument method, which specifies the scoring method to use (default is "outre").

The function returns an array of label quality scores, where each score is a continuous value between 0 and 1. A score of 1 indicates a clean label (likely correct), while a score of 0 indicates a dirty label (likely incorrect).

Ensure that the inputs are valid and then use the specified scoring method to calculate the label quality scores. The output is an array of scores with one score per example in the dataset.

Example usage:
```python
import numpy as np
from cleanlab.regression.rank import get_label_quality_scores

labels = np.array([1, 2, 3, 4])
predictions = np.array([2, 2, 5, 4.1])

label_quality_scores = get_label_quality_scores(labels, predictions)
print(label_quality_scores)
# Output: array([0.00323821, 0.33692597, 0.00191686, 0.33692597])
```"""
    labels, predictions = assert_valid_prediction_inputs(labels=labels, predictions=predictions, method=method)

    scoring_funcs: Dict[str, Callable] = {
        'residual': _get_residual_score_for_each_label,
        'outre': _get_outre_score_for_each_label,
    }
    scoring_func = scoring_funcs[method]
    label_quality_scores = scoring_func(labels, predictions)
    return label_quality_scores

def _get_residual_score_for_each_label(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Returns a residual label-quality score for each example.

    This is function to compute label-quality scores for regression datasets,
    where lower score indicate labels less likely to be correct.

    Residual based scores can work better for datasets where independent variables
    are based out of normal distribution.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    predictions: np.ndarray
        Predicted labels in the same format expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.

    """
    residual = predictions - labels
    label_quality_scores = np.exp(-abs(residual))
    return label_quality_scores

def _get_outre_score_for_each_label(labels: np.ndarray, predictions: np.ndarray, *, residual_scale: float=5, frac_neighbors: float=0.5, neighbor_metric: Optional[Union[str, Callable]]=None) -> np.ndarray:
    """Returns OUTRE based label-quality scores.

    This function computes label-quality scores for regression datasets,
    where a lower score indicates labels that are less likely to be correct.

    Parameters
    ----------
    labels: np.ndarray
        Labels in the same format as expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    predictions: np.ndarray
        Predicted labels in the same format as expected by the `~cleanlab.regression.rank.get_label_quality_scores` function.

    residual_scale: float, default = 5
        Multiplicative factor to adjust scale (standard deviation) of the residuals relative to the labels.

    frac_neighbors: float, default = 0.5
        Fraction of examples in dataset that should be considered as `n_neighbors` in the ``NearestNeighbors`` object used internally to assess outliers.

    neighbor_metric: Optional[str or callable], default = None
        The parameter is passed to sklearn NearestNeighbors. # TODO add reference to sklearn.NearestNeighbor?
        If None, the metric is chosen based on the number of features in the dataset.

    Returns
    -------
    label_quality_scores: np.ndarray
        Contains one score (between 0 and 1) per example.
        Lower scores indicate more likely mislabled examples.
    """
    residual = predictions - labels
    labels = (labels - labels.mean()) / (labels.std() + TINY_VALUE)
    residual = residual_scale * ((residual - residual.mean()) / (residual.std() + TINY_VALUE))
    features = np.array([labels, residual]).T
    neighbors = int(np.ceil(frac_neighbors * labels.shape[0]))
    neighbor_metric = neighbor_metric or decide_euclidean_metric(features)
    knn = features_to_knn(features, n_neighbors=neighbors, metric=neighbor_metric)
    ood = OutOfDistribution(params={'knn': knn})
    label_quality_scores = ood.score(features=features)
    return label_quality_scores
