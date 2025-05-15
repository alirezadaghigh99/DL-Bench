def prediction_is_close_to_threshold(
    prediction: Prediction,
    prediction_type: PredictionType,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
    minimum_objects_close_to_threshold: int,
) -> bool:
    if CLASSIFICATION_TASK not in prediction_type:
        return detections_are_close_to_threshold(
            prediction=prediction,
            selected_class_names=selected_class_names,
            threshold=threshold,
            epsilon=epsilon,
            minimum_objects_close_to_threshold=minimum_objects_close_to_threshold,
        )
    checker = multi_label_classification_prediction_is_close_to_threshold
    if "top" in prediction:
        checker = multi_class_classification_prediction_is_close_to_threshold
    return checker(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
        only_top_classes=only_top_classes,
    )