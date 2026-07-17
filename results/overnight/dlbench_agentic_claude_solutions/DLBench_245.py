from collections import Counter
from typing import Any, Sequence, Tuple
__all__ = ['ngrams', 'lcs', 'modified_precision']

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
    return Counter([tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)])

def lcs(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """
    Compute the length of the longest common subsequence in two sequence of items
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem

    Args:
        seq_a: first sequence of items
        seq_b: second sequence of items

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    m = len(seq_a)
    n = len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def modified_precision(references: Sequence[Sequence[Any]], candidate: Any, n: int) -> Tuple[int, int]:
    """
    Compute the modified precision

    .. math::
       p_{n} = \\frac{m_{n}}{l_{n}}

    where m_{n} is the number of matched n-grams between translation and references,
    l_{n} is the number of n-grams in the translation

    Args:
        references: list of references R
        candidate: translation T
        n: n-gram order

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    # Extract n-grams for the candidate
    counts = ngrams(candidate, n)

    # Extract n-grams for each of the references
    max_counts: Counter = Counter()
    for reference in references:
        max_counts |= ngrams(reference, n)

    # Clip the counts of the candidate n-grams by the maximum counts found in the references
    clipped_counts = counts & max_counts

    return sum(clipped_counts.values()), sum(counts.values())
