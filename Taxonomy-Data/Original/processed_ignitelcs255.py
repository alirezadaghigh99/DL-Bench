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