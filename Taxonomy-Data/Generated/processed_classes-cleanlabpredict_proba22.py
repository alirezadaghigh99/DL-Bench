import numpy as np
from sklearn.base import BaseEstimator

class CleanLearning(BaseEstimator):
    def __init__(
        self,
        clf=None,
        *,
        seed=None,
        cv_n_folds=5,
        converge_latent_estimates=False,
        pulearning=None,
        find_label_issues_kwargs={},
        label_quality_scores_kwargs={},
        verbose=False,
        low_memory=False,
    ):
        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.find_label_issues_kwargs = find_label_issues_kwargs
        self.label_quality_scores_kwargs = label_quality_scores_kwargs
        self.verbose = verbose
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        self.low_memory = low_memory

    def predict_proba(self, X, *args, **kwargs):
        # Ensure X is a two-dimensional array if the default classifier is used
        if self.clf is None:
            raise ValueError("No classifier has been set for CleanLearning.")
        
        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("Input data X must be a two-dimensional array.")
        
        # Call the predict_proba method of the wrapped classifier
        pred_probs = self.clf.predict_proba(X, *args, **kwargs)
        
        return pred_probs