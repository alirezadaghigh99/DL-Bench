import numpy as np

def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # Ensure y_true and y_score are numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Determine the positive label if not provided
    if pos_label is None:
        pos_label = 1 if y_true.dtype.kind in 'biu' else y_true.max()
    
    # Sort scores and corresponding true labels
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # If sample weights are provided, sort them as well
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        sample_weight = sample_weight[desc_score_indices]
    else:
        sample_weight = np.ones_like(y_true, dtype=float)
    
    # Calculate true positives and false positives
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum((y_true == pos_label) * sample_weight)[threshold_idxs]
    fps = np.cumsum((y_true != pos_label) * sample_weight)[threshold_idxs]
    
    # Total positives and negatives
    if sample_weight is not None:
        P = np.sum((y_true == pos_label) * sample_weight)
        N = np.sum((y_true != pos_label) * sample_weight)
    else:
        P = np.sum(y_true == pos_label)
        N = np.sum(y_true != pos_label)
    
    # Calculate FPR and FNR
    fpr = fps / N
    fnr = (P - tps) / P
    
    # Get thresholds
    thresholds = y_score[threshold_idxs]
    
    return fpr, fnr, thresholds

