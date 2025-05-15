import numpy as np
from sklearn.metrics import log_loss

def get_label_quality_ensemble_scores(labels, pred_probs_list, method='log_loss', adjust_pred_probs=False,
                                      weight_ensemble_members_by='uniform', custom_weights=None,
                                      log_loss_search_T_values=None, verbose=False):
    # Validate inputs
    if weight_ensemble_members_by == 'custom' and custom_weights is None:
        raise ValueError("Custom weights must be provided when using 'custom' weighting scheme.")
    
    if weight_ensemble_members_by == 'custom' and len(custom_weights) != len(pred_probs_list):
        raise ValueError("Length of custom weights must match the number of models in the ensemble.")
    
    # Initialize scores
    num_examples = labels.shape[0]
    num_models = len(pred_probs_list)
    scores = np.zeros((num_models, num_examples))
    
    # Calculate scores for each model
    for i, pred_probs in enumerate(pred_probs_list):
        if adjust_pred_probs:
            