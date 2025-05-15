import numpy as np
import pandas as pd

def find_overlapping_classes(labels=None, pred_probs=None, asymmetric=False, class_names=None, num_examples=None, joint=None, confident_joint=None):
    if joint is not None and num_examples is None:
        raise ValueError("If 'joint' is provided, 'num_examples' must also be provided.")
    
    if confident_joint is None:
        if labels is None or pred_probs is None:
            raise ValueError("If 'confident_joint' is not provided, both 'labels' and 'pred_probs' must be provided.")
        # Compute the confident joint from labels and pred_probs
        confident_joint = compute_confident_joint(labels, pred_probs)
    
    if joint is None:
        # Estimate the joint distribution from the confident joint
        joint = confident_joint / np.sum(confident_joint)
        num_examples = np.sum(confident_joint)
    
    K = joint.shape[0]
    overlapping_data = []

    for i in range(K):
        for j in range(i + 1, K):
            if asymmetric:
                num_overlap_ij = confident_joint[i, j]
                num_overlap_ji = confident_joint[j, i]
                joint_prob_ij = num_overlap_ij / num_examples
                joint_prob_ji = num_overlap_ji / num_examples
                overlapping_data.append((i, j, num_overlap_ij, joint_prob_ij))
                overlapping_data.append((j, i, num_overlap_ji, joint_prob_ji))
            else:
                num_overlap = confident_joint[i, j] + confident_joint[j, i]
                joint_prob = num_overlap / num_examples
                overlapping_data.append((i, j, num_overlap, joint_prob))
    
    df = pd.DataFrame(overlapping_data, columns=["Class Index A", "Class Index B", "Num Overlapping Examples", "Joint Probability"])
    df = df.sort_values(by="Joint Probability", ascending=False).reset_index(drop=True)
    
    if class_names is not None:
        df['Class Name A'] = df['Class Index A'].apply(lambda x: class_names[x])
        df['Class Name B'] = df['Class Index B'].apply(lambda x: class_names[x])
    
    return df

def compute_confident_joint(labels, pred_probs):
    # Placeholder for the actual computation of the confident joint
    # This function should be implemented based on the specific method used to compute the confident joint
    # For now, we'll assume it's a simple placeholder
    K = pred_probs.shape[1]
    confident_joint = np.zeros((K, K), dtype=int)
    for i, label in enumerate(labels):
        predicted_label = np.argmax(pred_probs[i])
        confident_joint[label, predicted_label] += 1
    return confident_joint