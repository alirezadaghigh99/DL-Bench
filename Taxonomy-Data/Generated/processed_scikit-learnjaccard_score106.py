import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

def jaccard_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    
    if average == 'binary':
        if len(labels) > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")
        pos_label = labels[1] if pos_label not in labels else pos_label
        y_true = (y_true == pos_label).astype(int)
        y_pred = (y_pred == pos_label).astype(int)
        labels = [pos_label]

    y_true_bin = label_binarize(y_true, classes=labels)
    y_pred_bin = label_binarize(y_pred, classes=labels)

    if average == 'samples':
        scores = np.array([_jaccard_index(y_true_bin[i], y_pred_bin[i]) for i in range(y_true_bin.shape[0])])
        return np.average(scores, weights=sample_weight)

    scores = np.array([_jaccard_index(y_true_bin[:, i], y_pred_bin[:, i]) for i in range(len(labels))])

    if average == 'micro':
        intersection = np.sum(np.logical_and(y_true_bin, y_pred_bin))
        union = np.sum(np.logical_or(y_true_bin, y_pred_bin))
        return _handle_zero_division(intersection, union, zero_division)

    if average == 'macro':
        return np.mean(scores)

    if average == 'weighted':
        support = np.sum(y_true_bin, axis=0)
        return np.average(scores, weights=support)

    return scores

def _jaccard_index(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    return _handle_zero_division(intersection, union, zero_division='warn')

def _handle_zero_division(intersection, union, zero_division):
    if union == 0:
        if zero_division == 'warn':
            print("Warning: Zero division encountered in Jaccard score calculation.")
            return 0.0
        return zero_division
    return intersection / union