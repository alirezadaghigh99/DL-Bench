import numpy as np
from scipy.sparse import csr_matrix

def get_num_labels(y):
    # Check if the input has a shape of 0
    if np.size(y) == 0:
        raise ValueError('Invalid labeling: Cannot contain 0 labels')
    
    # Check if y is an instance of csr_matrix
    if isinstance(y, csr_matrix):
        # Get the maximum index from the non-zero elements
        num_labels = y.indices.max() + 1
    else:
        # Assume y is a dense array-like structure
        num_labels = np.max(y) + 1
    
    return num_labels