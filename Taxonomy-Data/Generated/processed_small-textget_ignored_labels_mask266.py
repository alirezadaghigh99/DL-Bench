import numpy as np
from scipy.sparse import csr_matrix, issparse

def get_ignored_labels_mask(y, ignored_label_value):
    if issparse(y):
        # Handle the case where y is a csr_matrix
        mask = np.zeros(y.shape[0], dtype=bool)
        for i in range(y.shape[0]):
            # Check if any element in the row equals ignored_label_value
            row = y.getrow(i).toarray().flatten()
            if ignored_label_value in row:
                mask[i] = True
        return mask
    else:
        # Handle the case where y is a regular array
        y = np.asarray(y)  # Ensure y is a NumPy array
        return y == ignored_label_value

