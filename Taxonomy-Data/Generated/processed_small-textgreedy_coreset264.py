import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='euclidean', batch_size=1, normalized=False):
    # Normalize the data if required
    if normalized:
        x = normalize(x, axis=1)
    
    # Select the appropriate distance function
    if distance_metric == 'cosine':
        distance_func = cosine_distances
    elif distance_metric == 'euclidean':
        distance_func = euclidean_distances
    else:
        raise ValueError("Unsupported distance metric. Choose 'cosine' or 'euclidean'.")
    
    # Initialize the set of selected indices
    selected_indices = []
    
    # Create a mask for unlabeled data
    unlabeled_mask = np.ones(len(indices_unlabeled), dtype=bool)
    
    # Compute initial distances from labeled to unlabeled data
    labeled_data = x[indices_labeled]
    unlabeled_data = x[indices_unlabeled]
    distances = distance_func(labeled_data, unlabeled_data)
    
    # Main loop to select points for the coreset
    for _ in range(n):
        # Find the maximum distance for each unlabeled point to the labeled set
        min_distances = np.min(distances, axis=0)
        
        # Select the point with the maximum of these minimum distances
        max_dist_index = np.argmax(min_distances[unlabeled_mask])
        selected_index = indices_unlabeled[unlabeled_mask][max_dist_index]
        
        # Add the selected index to the coreset
        selected_indices.append(selected_index)
        
        # Update the mask to exclude the selected index
        unlabeled_mask[indices_unlabeled == selected_index] = False
        
        # Update the distances with the newly selected point
        new_point = x[selected_index].reshape(1, -1)
        new_distances = distance_func(new_point, unlabeled_data)
        distances = np.vstack((distances, new_distances))
    
    return np.array(selected_indices)

