from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
import numpy as np

class Scorer:
    def __init__(self, score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
        self.score_func = score_func
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.kwargs = kwargs

    def __call__(self, estimator, X, y_true):
        check_is_fitted(estimator)
        
        if self.needs_proba:
            y_pred = estimator.predict_proba(X)
        elif self.needs_threshold:
            if hasattr(estimator, "decision_function"):
                y_pred = estimator.decision_function(X)
            else:
                raise ValueError("Estimator does not have a decision_function method.")
        else:
            y_pred = estimator.predict(X)
        
        score = self.score_func(y_true, y_pred, **self.kwargs)
        
        if not self.greater_is_better:
            score = -score
        
        return score

def make_scorer(score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs):
    return Scorer(score_func, greater_is_better, needs_proba, needs_threshold, **kwargs)

