def get_num_labels(y):
    if y.shape[0] == 0:
        raise ValueError('Invalid labeling: Cannot contain 0 labels')

    if isinstance(y, csr_matrix):
        return np.max(y.indices) + 1
    else:
        return np.max(y) + 1