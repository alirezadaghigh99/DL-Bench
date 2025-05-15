def temp_scale_pred_probs(
    pred_probs: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Scales pred_probs by the given temperature factor. Temperature of <1 will sharpen the pred_probs while temperatures of >1 will smoothen it."""
    # clip pred_probs to prevent taking log of 0
    pred_probs = np.clip(pred_probs, a_min=SMALL_CONST, a_max=None)
    pred_probs = pred_probs / np.sum(pred_probs, axis=1)[:, np.newaxis]

    # apply temperate scale
    scaled_pred_probs = softmax(np.log(pred_probs), temperature=temp, axis=1, shift=False)
    scaled_pred_probs = (
        scaled_pred_probs / np.sum(scaled_pred_probs, axis=1)[:, np.newaxis]
    )  # normalize

    return scaled_pred_probs