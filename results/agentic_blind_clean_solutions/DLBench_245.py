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
    """Generate a Python function called modified_precision that computes the modified precision for a given list of references, a candidate translation, and an n-gram order. The function calculates the number of matched n-grams between the candidate translation and its references, and the total number of n-grams in the translation. The output is a tuple containing the sum of the clipped counts of the candidate and references, and the sum of the counts of the candidate."""
    candidate_ngrams = ngrams(candidate, n)
    total = sum(candidate_ngrams.values())
    if not candidate_ngrams:
        return 0, 0
    clipped = sum(
        min(count, max(ngrams(ref, n)[gram] for ref in references))
        for gram, count in candidate_ngrams.items()
    )
    return clipped, total
