import numpy as np

def exponential_moving_average(s, alpha=0.5, axis=0, **kwargs):
    """
    Calculate the exponential moving average (EMA) of an array of scores.

    Parameters:
    - s: np.ndarray, the input array of scores.
    - alpha: float, the forgetting factor (0 < alpha <= 1).
    - axis: int, the axis along which to compute the EMA.
    - **kwargs: additional keyword arguments (not used in this function).

    Returns:
    - s_ema: np.ndarray, the exponential moving average of the scores.
    """
    # Ensure the input is a numpy array
    s = np.asarray(s)
    
    # Initialize the EMA array with the same shape as s
    s_ema = np.zeros_like(s)
    
    # Move the specified axis to the front for easier iteration
    s = np.moveaxis(s, axis, 0)
    s_ema = np.moveaxis(s_ema, axis, 0)
    
    # Initialize the first EMA value
    s_ema[0] = s[0]
    
    # Compute the EMA for each subsequent score
    for t in range(1, s.shape[0]):
        s_ema[t] = alpha * s[t] + (1 - alpha) * s_ema[t - 1]
    
    # Move the axis back to its original position
    s_ema = np.moveaxis(s_ema, 0, axis)
    
    return s_ema

