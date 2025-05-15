import numpy as np

def combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state):
    # Set the random state for reproducibility
    rng = np.random.default_rng(random_state)
    
    n_samples, n_estimators = scores.shape
    combined_scores = np.zeros(n_samples)
    
    if method == 'static':
        # Static method: divide estimators into fixed subgroups
        bucket_size = n_estimators // n_buckets
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else n_estimators
            subgroup_scores = scores[:, start_idx:end_idx]
            combined_scores += np.mean(subgroup_scores, axis=1)
    elif method == 'dynamic':
        # Dynamic method: randomly assign estimators to subgroups
        for _ in range(n_buckets):
            if bootstrap_estimators:
                # Sample with replacement
                selected_indices = rng.choice(n_estimators, n_estimators, replace=True)
            else:
                # Sample without replacement
                selected_indices = rng.choice(n_estimators, n_estimators, replace=False)
            subgroup_scores = scores[:, selected_indices]
            combined_scores += np.mean(subgroup_scores, axis=1)
    else:
        raise ValueError("Method must be either 'static' or 'dynamic'")
    
    # Average the combined scores over the number of buckets
    combined_scores /= n_buckets
    return combined_scores

def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    """
    Maximization of Average ensemble method for combining multiple estimators.
    
    Parameters:
    - scores: numpy array of shape (n_samples, n_estimators)
    - n_buckets: int, number of subgroups to build (default is 5)
    - method: str, method to build subgroups ('static' or 'dynamic', default is 'static')
    - bootstrap_estimators: bool, whether estimators are drawn with replacement (default is False)
    - random_state: int, RandomState instance, or None, seed for random number generator (default is None)
    
    Returns:
    - combined_scores: numpy array of shape (n_samples,)
    """
    return combo_moa(scores, n_buckets, method, bootstrap_estimators, random_state)