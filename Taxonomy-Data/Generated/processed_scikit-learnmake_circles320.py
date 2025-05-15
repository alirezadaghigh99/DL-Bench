import numpy as np

def make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8):
    if isinstance(n_samples, int):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    elif isinstance(n_samples, tuple):
        if len(n_samples) != 2:
            raise ValueError("When a tuple, n_samples must have exactly two elements.")
        n_samples_out, n_samples_in = n_samples
    else:
        raise ValueError("n_samples must be an integer or a tuple of two integers.")
    
    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1.")
    
    rng = np.random.default_rng(random_state)
    
    # Generate points for the outer circle
    angles_out = rng.uniform(0, 2 * np.pi, n_samples_out)
    outer_circle = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    
    # Generate points for the inner circle
    angles_in = rng.uniform(0, 2 * np.pi, n_samples_in)
    inner_circle = factor * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    
    # Combine the points
    X = np.vstack((outer_circle, inner_circle))
    y = np.hstack((np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)))
    
    # Add noise if specified
    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)
    
    # Shuffle the dataset if specified
    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]
    
    return X, y

