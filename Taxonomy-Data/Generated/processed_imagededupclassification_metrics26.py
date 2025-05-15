def calculate_metrics(ground_truth, retrieved):
    def calculate_class_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1_score

    # Initialize counts for true positives, false positives, and false negatives
    tp_1, fp_1, fn_1 = 0, 0, 0
    tp_0, fp_0, fn_0 = 0, 0, 0

    # Iterate over each file in the ground truth
    for file, true_duplicates in ground_truth.items():
        retrieved_duplicates = retrieved.get(file, [])

        # Calculate true positives, false positives, and false negatives for class 1
        tp_1 += len(set(true_duplicates) & set(retrieved_duplicates))
        fp_1 += len(set(retrieved_duplicates) - set(true_duplicates))
        fn_1 += len(set(true_duplicates) - set(retrieved_duplicates))

        # Calculate true positives, false positives, and false negatives for class 0
        all_files = set(ground_truth.keys()) | set(retrieved.keys())
        non_duplicates = all_files - set(true_duplicates) - {file}
        retrieved_non_duplicates = all_files - set(retrieved_duplicates) - {file}

        tp_0 += len(non_duplicates & retrieved_non_duplicates)
        fp_0 += len(retrieved_non_duplicates - non_duplicates)
        fn_0 += len(non_duplicates - retrieved_non_duplicates)

    # Calculate metrics for class 1 (duplicates)
    precision_1, recall_1, f1_score_1 = calculate_class_metrics(tp_1, fp_1, fn_1)

    # Calculate metrics for class 0 (non-duplicates)
    precision_0, recall_0, f1_score_0 = calculate_class_metrics(tp_0, fp_0, fn_0)

    return {
        'class_1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1_score': f1_score_1
        },
        'class_0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1_score': f1_score_0
        }
    }

