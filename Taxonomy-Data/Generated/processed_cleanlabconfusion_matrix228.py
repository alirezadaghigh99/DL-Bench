import numpy as np

def confusion_matrix(true, pred):
    # Ensure that true and pred are numpy arrays
    true = np.asarray(true)
    pred = np.asarray(pred)
    
    # Check if the lengths of true and pred are the same
    if len(true) != len(pred):
        raise ValueError("The length of true and pred must be the same.")
    
    # Get the unique class labels
    labels = np.unique(true)
    
    # Initialize the confusion matrix with zeros
    num_classes = len(labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Create a mapping from label to index
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    # Populate the confusion matrix
    for t, p in zip(true, pred):
        true_index = label_to_index[t]
        pred_index = label_to_index[p]
        conf_matrix[true_index, pred_index] += 1
    
    return conf_matrix

