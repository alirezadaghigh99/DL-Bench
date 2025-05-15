def get_ignored_labels_mask(y, ignored_label_value):
    if isinstance(y, csr_matrix):
        return np.array([(row.toarray() == ignored_label_value).any() for row in y])
    else:
        return y == np.array([ignored_label_value])