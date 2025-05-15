import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.utils import check_matplotlib_support
from sklearn.calibration import CalibrationDisplay

def from_estimator(estimator, X, y, n_bins=10, strategy='uniform', pos_label=None, name=None, ref_line=True, ax=None, **kwargs):
    """
    Plot a calibration curve using a binary classifier and data.

    Parameters:
    - estimator: Fitted binary classifier.
    - X: Input values.
    - y: Binary target values.
    - n_bins: Number of bins for discretization.
    - strategy: Strategy for bin width ('uniform' or 'quantile').
    - pos_label: Positive class label.
    - name: Label for the curve.
    - ref_line: Whether to plot a reference line.
    - ax: Matplotlib axes.
    - **kwargs: Additional keyword arguments for plotting.

    Returns:
    - CalibrationDisplay: Object that stores computed values.
    """
    check_matplotlib_support('from_estimator')

    # Get the predicted probabilities for the positive class
    if hasattr(estimator, "predict_proba"):
        prob_pos = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        prob_pos = estimator.decision_function(X)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    else:
        raise ValueError("Estimator should have either a predict_proba or decision_function method.")

    # Compute the calibration curve
    prob_true, prob_pred = calibration_curve(y, prob_pos, n_bins=n_bins, strategy=strategy, pos_label=pos_label)

    # Create a plot if no axes are provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the calibration curve
    line_label = name if name is not None else estimator.__class__.__name__
    ax.plot(prob_pred, prob_true, marker='o', label=line_label, **kwargs)

    # Plot the reference line
    if ref_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')

    # Set plot labels and limits
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')

    # Return a CalibrationDisplay object
    return CalibrationDisplay(prob_true=prob_true, prob_pred=prob_pred, estimator_name=line_label, ax=ax)

