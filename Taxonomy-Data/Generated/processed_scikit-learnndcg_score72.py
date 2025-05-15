import numpy as np

def dcg_score(y_true, y_score, k=None, ignore_ties=False):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if ignore_ties:
        gains = y_true
    else:
        gains = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=None, sample_weight=None, ignore_ties=False):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0])
    else:
        sample_weight = np.asarray(sample_weight)

    scores = []
    for i in range(y_true.shape[0]):
        actual = y_true[i]
        predicted = y_score[i]

        best_dcg = dcg_score(actual, actual, k, ignore_ties)
        actual_dcg = dcg_score(actual, predicted, k, ignore_ties)

        if best_dcg == 0:
            score = 0.0
        else:
            score = actual_dcg / best_dcg

        scores.append(score * sample_weight[i])

    return np.sum(scores) / np.sum(sample_weight)

