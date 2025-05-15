from sklearn.metrics import get_scorer as sklearn_get_scorer
from copy import deepcopy

def get_scorer(scoring):
    """
    Retrieve a scorer based on the input scoring method.

    Parameters:
    scoring (str, callable, or None): The scoring method to retrieve. It can be:
        - A string representing the name of a scoring method available in sklearn.
        - A callable that takes (estimator, X, y) as parameters and returns a score.
        - None, in which case the function returns None.

    Returns:
    callable or None: A scorer object based on the input scoring method, or None if the input is None.

    Raises:
    ValueError: If the input scoring value is not a valid string, callable, or None.

    Examples:
    >>> from sklearn.metrics import accuracy_score
    >>> scorer = get_scorer('accuracy')
    >>> print(scorer)
    <function _passthrough_scorer at 0x...>

    >>> custom_scorer = lambda estimator, X, y: accuracy_score(y, estimator.predict(X))
    >>> scorer = get_scorer(custom_scorer)
    >>> print(scorer)
    <function <lambda> at 0x...>

    >>> scorer = get_scorer(None)
    >>> print(scorer)
    None

    >>> get_scorer(123)  # Raises ValueError
    """
    if isinstance(scoring, str):
        try:
            return deepcopy(sklearn_get_scorer(scoring))
        except ValueError as e:
            raise ValueError(f"Invalid scoring string: {scoring}") from e
    elif callable(scoring):
        return scoring
    elif scoring is None:
        return None
    else:
        raise ValueError("Scoring must be a string, a callable, or None.")

