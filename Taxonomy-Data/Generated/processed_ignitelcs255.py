def lcs(seq_a, seq_b):
    """
    Compute the length of the longest common subsequence in two sequences.

    Args:
        seq_a: first sequence of items
        seq_b: second sequence of items

    Returns:
        The length of the longest common subsequence
    """
    # Get the lengths of the two sequences
    len_a = len(seq_a)
    len_b = len(seq_b)

    # Create a 2D array to store the lengths of LCS
    # Initialize with zeros
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    # Fill the dp array
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # The length of the longest common subsequence is in the bottom-right cell
    return dp[len_a][len_b]

