class Prediction:
    def __init__(self, detections):
        """
        Initialize the Prediction object with a list of detections.
        Each detection is a dictionary with 'class_name' and 'confidence' keys.
        """
        self.detections = detections

def count_detections_close_to_threshold(prediction, selected_class_names, threshold, epsilon):
    """
    Count the number of detections whose confidence is within epsilon of the threshold
    and belong to the selected class names.
    """
    count = 0
    for detection in prediction.detections:
        class_name = detection.get('class_name')
        confidence = detection.get('confidence')
        
        if class_name in selected_class_names and abs(confidence - threshold) <= epsilon:
            count += 1
    
    return count

def detections_are_close_to_threshold(prediction, selected_class_names, threshold, epsilon, min_objects_close):
    """
    Determine if the number of detections close to the threshold is greater than or equal
    to the minimum number of objects specified.
    """
    count = count_detections_close_to_threshold(prediction, selected_class_names, threshold, epsilon)
    return count >= min_objects_close

