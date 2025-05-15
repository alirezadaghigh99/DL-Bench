def classification_metrics(ground_truth: Dict, retrieved: Dict) -> np.ndarray:
    """
    Given ground truth dictionary and retrieved dictionary, return per class precision, recall and f1 score. Class 1 is
    assigned to duplicate file pairs while class 0 is for non-duplicate file pairs.

    Args:
        ground_truth: A dictionary representing ground truth with filenames as key and a list of duplicate filenames
        as value.
        retrieved: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved
        duplicate filenames as value.

    Returns:
        Dictionary of precision, recall and f1 score for both classes.
    """
    all_pairs = _make_all_unique_possible_pairs(ground_truth)
    ground_truth_duplicate_pairs, retrieved_duplicate_pairs = _make_positive_duplicate_pairs(
        ground_truth, retrieved
    )
    y_true, y_pred = _prepare_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )
    logger.info(classification_report(y_true, y_pred))
    prec_rec_fscore_support = dict(
        zip(
            ('precision', 'recall', 'f1_score', 'support'),
            precision_recall_fscore_support(y_true, y_pred),
        )
    )
    return prec_rec_fscore_support