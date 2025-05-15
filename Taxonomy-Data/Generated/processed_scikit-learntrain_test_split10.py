import numpy as np

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    # Input validation
    if not arrays:
        raise ValueError("At least one array is required as input.")
    
    n_samples = len(arrays[0])
    for array in arrays:
        if len(array) != n_samples:
            raise ValueError("All input arrays must have the same number of samples.")
    
    if test_size is None and train_size is None:
        test_size = 0.25
    elif test_size is not None and train_size is not None:
        raise ValueError("Specify either test_size or train_size, not both.")
    
    if test_size is not None:
        if isinstance(test_size, float):
            test_size = int(n_samples * test_size)
        elif isinstance(test_size, int):
            if test_size >= n_samples or test_size <= 0:
                raise ValueError("test_size must be between 0 and the number of samples.")
    elif train_size is not None:
        if isinstance(train_size, float):
            train_size = int(n_samples * train_size)
        elif isinstance(train_size, int):
            if train_size >= n_samples or train_size <= 0:
                raise ValueError("train_size must be between 0 and the number of samples.")
        test_size = n_samples - train_size
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    
    if shuffle:
        if stratify is not None:
            unique_classes, y_indices = np.unique(stratify, return_inverse=True)
            class_counts = np.bincount(y_indices)
            test_indices = []
            for class_index, class_count in enumerate(class_counts):
                class_indices = np.where(y_indices == class_index)[0]
                np.random.shuffle(class_indices)
                n_test = int(np.floor(test_size * class_count / n_samples))
                test_indices.extend(class_indices[:n_test])
            test_indices = np.array(test_indices)
            train_indices = np.setdiff1d(indices, test_indices)
        else:
            np.random.shuffle(indices)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
    else:
        if stratify is not None:
            raise ValueError("Stratification requires shuffling.")
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
    
    result = []
    for array in arrays:
        result.append(array[train_indices])
        result.append(array[test_indices])
    
    return result

