def normalized_mse(ref_outputs: List[np.ndarray], approx_outputs: List[np.ndarray]) -> float:
    """
    Calculates normalized mean square error between `ref_outputs` and `approx_outputs`.
    The normalized mean square error is defined as

    NMSE(x_ref, x_approx) = MSE(x_ref, x_approx) / MSE(x_ref, 0)

    :param ref_outputs: Reference outputs.
    :param approx_outputs: Approximate outputs.
    :return: The normalized mean square error between `ref_outputs` and `approx_outputs`.
    """
    metrics = []
    for x_ref, x_approx in zip(ref_outputs, approx_outputs):
        error_flattened = (x_ref - x_approx).flatten()
        x_ref_flattened = x_ref.flatten()
        nmse = np.dot(error_flattened, error_flattened) / np.dot(x_ref_flattened, x_ref_flattened)
        metrics.append(nmse)
    nmse = sum(metrics) / len(metrics)
    return nmse