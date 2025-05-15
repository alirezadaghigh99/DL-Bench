import numpy as np

def compute_confident_joint(labels, pred_probs, thresholds=None, calibrate=False, 
                            multi_label=False, return_indices_of_off_diagonals=False):
    """
    Estimates the confident counts of latent true vs observed noisy labels.

    Parameters:
    - labels: An array or list of class labels for each example in the dataset.
    - pred_probs: An array of model-predicted class probabilities for each example in the dataset.
    - thresholds: An optional array of per-class threshold probabilities.
    - calibrate: A boolean flag indicating whether to calibrate the confident joint estimate.
    - multi_label: An optional boolean flag indicating if the dataset is multi-label classification.
    - return_indices_of_off_diagonals: An optional boolean flag indicating whether to return indices of examples counted in off-diagonals of the confident joint.

    Returns:
    - A numpy array representing counts of examples for which we are confident about their given and true label.
    """
    labels = np.array(labels)
    pred_probs = np.array(pred_probs)
    
    num_classes = pred_probs.shape[1]
    
    if thresholds is None:
        thresholds = np.full(num_classes, 0.5)
    else:
        thresholds = np.array(thresholds)
    
    if multi_label:
        confident_joint = np.zeros((num_classes, 2, 2), dtype=int)
    else:
        confident_joint = np.zeros((num_classes, num_classes), dtype=int)
    
    if return_indices_of_off_diagonals:
        off_diagonal_indices = []

    for i, (label, probs) in enumerate(zip(labels, pred_probs)):
        if multi_label:
            for class_idx in range(num_classes):
                if probs[class_idx] >= thresholds[class_idx]:
                    confident_joint[class_idx, int(label[class_idx]), 1] += 1
                else:
                    confident_joint[class_idx, int(label[class_idx]), 0] += 1
        else:
            predicted_label = np.argmax(probs)
            if probs[predicted_label] >= thresholds[predicted_label]:
                confident_joint[label, predicted_label] += 1
                if return_indices_of_off_diagonals and label != predicted_label:
                    off_diagonal_indices.append(i)
    
    if calibrate:
        # Placeholder for calibration logic
        # Implement calibration logic if needed
        pass
    
    if return_indices_of_off_diagonals:
        return confident_joint, off_diagonal_indices
    else:
        return confident_joint

