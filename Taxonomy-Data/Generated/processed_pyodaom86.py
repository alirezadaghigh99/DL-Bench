import numpy as np

def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None):
    """
    Implements the Average of Maximum (AOM) ensemble method for combining multiple estimators.

    Parameters:
    - scores: numpy array of shape (n_samples, n_estimators) representing the score matrix outputted from various estimators
    - n_buckets: integer specifying the number of subgroups to build (default value is 5)
    - method: string specifying the method for building subgroups ('static' or 'dynamic', default value is 'static')
    - bootstrap_estimators: boolean indicating whether estimators are drawn with replacement (default value is False)
    - random_state: integer, RandomState instance, or None specifying the seed for the random number generator (default value is None)

    Returns:
    - combined_scores: numpy array of shape (n_samples,) representing the combined outlier scores.
    """
    np.random.seed(random_state)
    n_samples, n_estimators = scores.shape

    if method not in ['static', 'dynamic']:
        raise ValueError("Method must be either 'static' or 'dynamic'.")

    if n_buckets > n_estimators:
        raise ValueError("Number of buckets cannot be greater than the number of estimators.")

    # Initialize the combined scores array
    combined_scores = np.zeros(n_samples)

    # Determine the size of each bucket
    if method == 'static':
        bucket_size = n_estimators // n_buckets
        remainder = n_estimators % n_buckets
    else:  # dynamic
        bucket_sizes = np.random.randint(1, n_estimators, size=n_buckets)
        bucket_sizes = (bucket_sizes / bucket_sizes.sum() * n_estimators).astype(int)
        bucket_sizes[-1] += n_estimators - bucket_sizes.sum()  # Adjust to ensure total equals n_estimators

    # Create buckets and calculate the maximum score for each bucket
    start_idx = 0
    for i in range(n_buckets):
        if method == 'static':
            end_idx = start_idx + bucket_size + (1 if i < remainder else 0)
        else:  # dynamic
            end_idx = start_idx + bucket_sizes[i]

        if bootstrap_estimators:
            bucket_indices = np.random.choice(n_estimators, end_idx - start_idx, replace=True)
        else:
            bucket_indices = np.arange(start_idx, end_idx)

        # Calculate the maximum score for each sample in the current bucket
        max_scores = np.max(scores[:, bucket_indices], axis=1)
        combined_scores += max_scores

        start_idx = end_idx

    # Average the maximum scores from each bucket
    combined_scores /= n_buckets

    return combined_scores