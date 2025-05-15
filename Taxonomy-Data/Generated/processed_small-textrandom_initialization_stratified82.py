import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def stratified_sampling(y, n_samples):
    """Perform stratified sampling for single-label data."""
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(unique_classes)
    
    # Calculate the number of samples per class
    samples_per_class = np.bincount(y_indices, minlength=n_classes)
    samples_per_class = np.floor(samples_per_class * (n_samples / len(y))).astype(int)
    
    indices = []
    for class_index in range(n_classes):
        class_indices = np.where(y_indices == class_index)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class[class_index], replace=False)
        indices.extend(selected_indices)
    
    return np.array(indices)

def multilabel_stratified_subsets_sampling(y, n_samples):
    """Perform stratified sampling for multi-label data using labelsets."""
    # Convert csr_matrix to dense array for easier manipulation
    y_dense = y.toarray()
    
    # Create a unique labelset for each sample
    labelsets = [tuple(row) for row in y_dense]
    unique_labelsets, labelset_indices = np.unique(labelsets, return_inverse=True, axis=0)
    
    n_labelsets = len(unique_labelsets)
    
    # Calculate the number of samples per labelset
    samples_per_labelset = np.bincount(labelset_indices, minlength=n_labelsets)
    samples_per_labelset = np.floor(samples_per_labelset * (n_samples / len(y_dense))).astype(int)
    
    indices = []
    for labelset_index in range(n_labelsets):
        labelset_indices = np.where(labelset_indices == labelset_index)[0]
        selected_indices = np.random.choice(labelset_indices, samples_per_labelset[labelset_index], replace=False)
        indices.extend(selected_indices)
    
    return np.array(indices)

def random_initialization_stratified(y, n_samples=10, multilabel_strategy='labelsets'):
    if isinstance(y, csr_matrix):
        if multilabel_strategy == 'labelsets':
            return multilabel_stratified_subsets_sampling(y, n_samples)
        else:
            raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
    else:
        return stratified_sampling(y, n_samples)

