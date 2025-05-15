def detections_are_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    minimum_objects_close_to_threshold: int,
) -> bool:
    detections_close_to_threshold = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
    )
    return detections_close_to_threshold >= minimum_objects_close_to_threshold