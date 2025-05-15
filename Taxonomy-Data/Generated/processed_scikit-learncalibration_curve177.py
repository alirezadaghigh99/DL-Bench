import numpy as np
from sklearn.utils import column_or_1d
from sklearn.utils.validation import _deprecate_positional_args

@_deprecate_positional_args
def calibration_curve(y_true, y_prob, *, pos_label=None, n_bins=5, strategy='uniform'):
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if pos_label is None:
        pos_label = 1

    # Ensure y_true is binary
    y_true = (y_true == pos_label)

    if strategy not in ['uniform', 'quantile']:
        raise ValueError("Invalid value for strategy: {}. "
                         "Valid options are 'uniform' or 'quantile'.".format(strategy))

    if strategy == 'uniform':
        bins = np.linspace(0., 1., n_bins + 1)
    elif strategy == 'quantile':
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    binids = np.digitize(y_prob, bins) - 1

    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)

    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_true[i] = np.mean(y_true[mask])
            prob_pred[i] = np.mean(y_prob[mask])
        else:
            prob_true[i] = np.nan
            prob_pred[i] = np.nan

    # Remove bins with no samples
    mask = ~np.isnan(prob_true)
    prob_true = prob_true[mask]
    prob_pred = prob_pred[mask]

    return prob_true, prob_pred

