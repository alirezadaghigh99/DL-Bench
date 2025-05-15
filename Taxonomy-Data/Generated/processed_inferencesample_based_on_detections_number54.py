import numpy as np
import random

def count_detections_close_to_threshold(prediction, selected_class_names, threshold=0.5):
    # This is a placeholder function. You should replace it with the actual implementation.
    # For demonstration, let's assume it returns the number of detections close to the threshold.
    count = 0
    for detection in prediction['detections']:
        if detection['class_name'] in selected_class_names and abs(detection['score'] - threshold) < 0.1:
            count += 1
    return count

def sample_based_on_detections_number(image, prediction, prediction_type, more_than=None, less_than=None, selected_class_names=set(), probability=1.0):
    # Define eligible prediction types
    eligible_types = {'type1', 'type2', 'type3'}  # Replace with actual eligible types

    # Check if prediction is a stub or if prediction type is not eligible
    if prediction is None or prediction_type not in eligible_types:
        return False

    # Calculate the number of detections close to the threshold
    num_detections = count_detections_close_to_threshold(prediction, selected_class_names)

    # Check if the number of detections falls within the specified range
    if (more_than is not None and num_detections <= more_than) or (less_than is not None and num_detections >= less_than):
        return False

    # Return True with a probability determined by the random number generator
    return random.random() < probability

