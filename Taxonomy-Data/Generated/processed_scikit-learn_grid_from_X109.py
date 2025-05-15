import numpy as np
from itertools import product

def _grid_from_X(X, percentiles=(0.05, 0.95), is_categorical=None, grid_resolution=100):
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    if is_categorical is None:
        is_categorical = [False] * n_features
    
    values = []
    
    for i in range(n_features):
        if is_categorical[i]:
            # Use unique values for categorical features
            unique_values = np.unique(X[:, i])
            values.append(unique_values)
        else:
            # Calculate percentiles for numerical features
            lower, upper = np.percentile(X[:, i], [percentiles[0] * 100, percentiles[1] * 100])
            unique_values = np.unique(X[:, i])
            
            if len(unique_values) <= grid_resolution:
                # Use unique values if they are fewer than grid_resolution
                values.append(unique_values)
            else:
                # Generate grid_resolution equally spaced points between percentiles
                values.append(np.linspace(lower, upper, grid_resolution))
    
    # Generate the cartesian product of all values
    grid = np.array(list(product(*values)))
    
    return grid, values

