def is_equal_tf(
    x: Union[tf.Tensor, np.ndarray, List],
    y: Union[tf.Tensor, np.ndarray, List],
    atol: float = EPS,
) -> bool:
    """
    Check if two tf tensors are identical within a tolerance.

    :param x:
    :param y:
    :param atol: error margin
    :return: return true if two tf tensors are nearly equal
    """
    x = tf.cast(x, dtype=tf.float32).numpy()
    y = tf.cast(y, dtype=tf.float32).numpy()
    return is_equal_np(x=x, y=y, atol=atol)