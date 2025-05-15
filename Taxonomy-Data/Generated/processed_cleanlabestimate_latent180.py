import numpy as np

def estimate_latent(confident_joint, labels, py_method="cnt", converge_latent_estimates=False):
    """
    Computes the latent prior p(y), the noise matrix P(labels|y), and the inverse noise matrix P(y|labels)
    from the confident_joint.

    Parameters
    ----------
    confident_joint : np.ndarray
        An array of shape (K, K) representing the confident joint.

    labels : np.ndarray
        A 1D array of shape (N,) containing class labels.

    py_method : {"cnt", "eqn", "marginal", "marginal_ps"}, default="cnt"
        Method to compute the latent prior p(true_label=k).

    converge_latent_estimates : bool, optional
        If True, forces numerical consistency of estimates.

    Returns
    ------
    tuple
        A tuple containing (py, noise_matrix, inv_noise_matrix).
    """
    K = confident_joint.shape[0]
    
    # Compute the latent prior p(y)
    if py_method == "cnt":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    elif py_method == "eqn":
        py = np.linalg.solve(confident_joint.T, np.sum(confident_joint, axis=1))
    elif py_method == "marginal":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    elif py_method == "marginal_ps":
        py = np.sum(confident_joint, axis=0) / np.sum(confident_joint)
    else:
        raise ValueError(f"Invalid py_method: {py_method}")

    # Compute the noise matrix P(labels|y)
    noise_matrix = confident_joint / np.sum(confident_joint, axis=0, keepdims=True)

    # Compute the inverse noise matrix P(y|labels)
    inv_noise_matrix = confident_joint / np.sum(confident_joint, axis=1, keepdims=True)

    if converge_latent_estimates:
        # Iteratively adjust estimates to ensure consistency
        for _ in range(10):  # Arbitrary number of iterations for convergence
            py = np.sum(confident_joint * inv_noise_matrix, axis=0) / np.sum(confident_joint)
            noise_matrix = confident_joint / np.sum(confident_joint, axis=0, keepdims=True)
            inv_noise_matrix = confident_joint / np.sum(confident_joint, axis=1, keepdims=True)

    return py, noise_matrix, inv_noise_matrix