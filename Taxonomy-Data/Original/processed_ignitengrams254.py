def ngrams(sequence: Sequence[Any], n: int) -> Counter:
    """
    Generate the ngrams from a sequence of items

    Args:
        sequence: sequence of items
        n: n-gram order

    Returns:
        A counter of ngram objects

    .. versionadded:: 0.4.5
    """
    return Counter([tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)])