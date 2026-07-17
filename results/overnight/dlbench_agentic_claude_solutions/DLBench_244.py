"""
Helper methods that are useful for benchmarking cleanlab’s core algorithms.
These methods introduce synthetic noise into the labels of a classification dataset.
Specifically, this module provides methods for generating valid noise matrices (for which learning with noise is possible),
generating noisy labels given a noise matrix, generating valid noise matrices with a specific trace value, and more.
"""
from typing import Optional
import numpy as np
from cleanlab.internal.util import value_counts
from cleanlab.internal.constants import FLOATING_POINT_COMPARISON

def noise_matrix_is_valid(noise_matrix, py, *, verbose=False) -> bool:
    """Given a prior `py` representing ``p(true_label=k)``, checks if the given `noise_matrix` is a
    learnable matrix. Learnability means that it is possible to achieve
    better than random performance, on average, for the amount of noise in
    `noise_matrix`.

    Parameters
    ----------
    noise_matrix : np.ndarray
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    py : np.ndarray
      An array of shape ``(K,)`` representing the fraction (prior probability)
      of each true class label, ``P(true_label = k)``.

    Returns
    -------
    is_valid : bool
      Whether the noise matrix is a learnable matrix.
    """
    K = len(py)
    N = float(10000)
    ps = np.dot(noise_matrix, py)
    joint_noise = np.multiply(noise_matrix, py)
    if not abs(joint_noise.sum() - 1.0) < FLOATING_POINT_COMPARISON:
        return False
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print('E1E2/C', round(E1 * E2 / C), 'E1', round(E1), 'E2', round(E2), 'C', round(C), '|', round(E1 * E2 / C + E1 + E2 + C), '|', round(E1 * E2 / C), '<', round(O))
            print(round(ps[i] * py[i]), '<', round(joint_noise[i][i]), ':', ps[i] * py[i] < joint_noise[i][i])
        if not ps[i] * py[i] < joint_noise[i][i]:
            return False
    return True

def generate_noisy_labels(true_labels, noise_matrix) -> np.ndarray:
    """Generates noisy `labels` from perfect labels `true_labels`,
    "exactly" yielding the provided `noise_matrix` between `labels` and `true_labels`.

    Below we provide a for loop implementation of what this function does.
    We do not use this implementation as it is not a fast algorithm, but
    it explains as Python pseudocode what is happening in this function.

    Parameters
    ----------
    true_labels : np.ndarray
      An array of shape ``(N,)`` representing perfect labels, without any
      noise. Contains K distinct natural number classes, 0, 1, ..., K-1.

    noise_matrix : np.ndarray
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    Returns
    -------
    labels : np.ndarray
      An array of shape ``(N,)`` of noisy labels.

    Examples
    --------

    .. code:: python

        # Generate labels
        count_joint = (noise_matrix * py * len(y)).round().astype(int)
        labels = np.ndarray(y)
        for k_s in range(K):
            for k_y in range(K):
                if k_s != k_y:
                    idx_flip = np.where((labels==k_y)&(true_label==k_y))[0]
                    if len(idx_flip): # pragma: no cover
                        labels[np.random.choice(
                            idx_flip,
                            count_joint[k_s][k_y],
                            replace=False,
                        )] = k_s
    """
    true_labels = np.asarray(true_labels)
    K = len(noise_matrix)
    py = value_counts(true_labels) / float(len(true_labels))
    count_joint = (noise_matrix * py * len(true_labels)).astype(int)
    np.fill_diagonal(count_joint, 0)
    labels = np.array(true_labels)
    for k in range(K):
        labels_per_class = np.where(count_joint[:, k] != 0)[0]
        label_counts = count_joint[labels_per_class, k]
        noise = [labels_per_class[i] for (i, c) in enumerate(label_counts) for z in range(c)]
        idx_flip = np.where((labels == k) & (true_labels == k))[0]
        if len(idx_flip) and len(noise) and (len(idx_flip) >= len(noise)):
            labels[np.random.choice(idx_flip, len(noise), replace=False)] = noise
    return labels

def generate_noise_matrix_from_trace(K, trace, *, max_trace_prob=1.0, min_trace_prob=1e-05, max_noise_rate=1 - 1e-05, min_noise_rate=0.0, valid_noise_matrix=True, py=None, frac_zero_noise_rates=0.0, seed=0, max_iter=10000) -> Optional[np.ndarray]:
    """Create a Python function called generate_noise_matrix_from_trace that Generates a ``K x K`` noise matrix ``P(label=k_s|true_label=k_y)`` with
    ``np.sum(np.diagonal(noise_matrix))`` equal to the given `trace`.

    Parameters
    ----------
    K : int
      Creates a noise matrix of shape ``(K, K)``. Implies there are
      K classes for learning with noisy labels.

    trace : float
      Sum of diagonal entries of array of random probabilities returned.

    max_trace_prob : float
      Maximum probability of any entry in the trace of the return matrix.

    min_trace_prob : float
      Minimum probability of any entry in the trace of the return matrix.

    max_noise_rate : float
      Maximum noise_rate (non-diagonal entry) in the returned np.ndarray.

    min_noise_rate : float
      Minimum noise_rate (non-diagonal entry) in the returned np.ndarray.

    valid_noise_matrix : bool, default=True
      If ``True``, returns a matrix having all necessary conditions for
      learning with noisy labels. In particular, ``p(true_label=k)p(label=k) < p(true_label=k,label=k)``
      is satisfied. This requires that ``trace > 1``.

    py : np.ndarray
      An array of shape ``(K,)`` representing the fraction (prior probability) of each true class label, ``P(true_label = k)``.
      This argument is **required** when ``valid_noise_matrix=True``.

    frac_zero_noise_rates : float
      The fraction of the ``n*(n-1)`` noise rates
      that will be set to 0. Note that if you set a high trace, it may be
      impossible to also have a low fraction of zero noise rates without
      forcing all non-1 diagonal values. Instead, when this happens we only
      guarantee to produce a noise matrix with `frac_zero_noise_rates` *or
      higher*. The opposite occurs with a small trace.

    seed : int
      Seeds the random number generator for numpy.

    max_iter : int, default=10000
      The max number of tries to produce a valid matrix before returning ``None``.

    Returns
    -------
    noise_matrix : np.ndarray or None
      An array of shape ``(K, K)`` representing the noise matrix ``P(label=k_s|true_label=k_y)`` with `trace`
      equal to ``np.sum(np.diagonal(noise_matrix))``. This a conditional probability matrix and a
      left stochastic matrix. Returns ``None`` if `max_iter` is exceeded."""
    if valid_noise_matrix and trace <= 1:
        raise ValueError('trace = {}. trace must be > 1 for a valid noise matrix to be returned (valid_noise_matrix == True).'.format(trace))
    if valid_noise_matrix and py is None:
        raise ValueError('py must be provided (not None) when valid_noise_matrix == True.')

    np.random.seed(seed)

    num_off_diag = K * (K - 1)
    num_zeros = int(round(frac_zero_noise_rates * num_off_diag))

    for _ in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))
        diag = generate_n_rand_probabilities_that_sum_to_m(K, trace, max_prob=max_trace_prob, min_prob=min_trace_prob)
        np.fill_diagonal(noise_matrix, diag)

        zeros_per_col = randomly_distribute_N_balls_into_K_bins(num_zeros, K, max_balls_per_bin=K - 1)

        try:
            for col in range(K):
                off_diag_rows = [row for row in range(K) if row != col]
                n_zero = int(zeros_per_col[col])
                remaining_mass = 1.0 - diag[col]
                nonzero_rows = off_diag_rows
                if n_zero > 0:
                    zero_rows = set(np.random.choice(off_diag_rows, size=n_zero, replace=False))
                    nonzero_rows = [row for row in off_diag_rows if row not in zero_rows]
                if nonzero_rows:
                    probs = generate_n_rand_probabilities_that_sum_to_m(len(nonzero_rows), remaining_mass, max_prob=max_noise_rate, min_prob=min_noise_rate)
                    for row, prob in zip(nonzero_rows, probs):
                        noise_matrix[row][col] = prob
                elif remaining_mass > FLOATING_POINT_COMPARISON:
                    raise ValueError('Cannot zero out every off-diagonal entry in a column that still has probability mass remaining.')
        except ValueError:
            continue

        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            return noise_matrix

    return None

def generate_n_rand_probabilities_that_sum_to_m(n, m, *, max_prob=1.0, min_prob=0.0) -> np.ndarray:
    """
    Generates `n` random probabilities that sum to `m`.

    When ``min_prob=0`` and ``max_prob = 1.0``, use
    ``np.random.dirichlet(np.ones(n))*m`` instead.

    Parameters
    ----------
    n : int
      Length of array of random probabilities to be returned.

    m : float
      Sum of array of random probabilities that is returned.

    max_prob : float, default=1.0
      Maximum probability of any entry in the returned array. Must be between 0 and 1.

    min_prob : float, default=0.0
      Minimum probability of any entry in the returned array. Must be between 0 and 1.

    Returns
    -------
    probabilities : np.ndarray
      An array of probabilities.
    """
    if n == 0:
        return np.array([])
    if max_prob + FLOATING_POINT_COMPARISON < m / float(n):
        raise ValueError('max_prob must be greater or equal to m / n, but ' + 'max_prob = ' + str(max_prob) + ', m = ' + str(m) + ', n = ' + str(n) + ', m / n = ' + str(m / float(n)))
    if min_prob > (m + FLOATING_POINT_COMPARISON) / float(n):
        raise ValueError('min_prob must be less or equal to m / n, but ' + 'max_prob = ' + str(max_prob) + ', m = ' + str(m) + ', n = ' + str(n) + ', m / n = ' + str(m / float(n)))
    result = np.random.dirichlet(np.ones(n)) * m
    min_val = min(result)
    max_val = max(result)
    while max_val > max_prob + FLOATING_POINT_COMPARISON:
        new_min = min_val + (max_val - max_prob)
        adjustment = (max_prob - new_min) * np.random.rand()
        result[np.argmin(result)] = new_min + adjustment
        result[np.argmax(result)] = max_prob - adjustment
        min_val = min(result)
        max_val = max(result)
    min_val = min(result)
    max_val = max(result)
    while min_val < min_prob - FLOATING_POINT_COMPARISON:
        min_val = min(result)
        max_val = max(result)
        new_max = max_val - (min_prob - min_val)
        adjustment = (new_max - min_prob) * np.random.rand()
        result[np.argmax(result)] = new_max - adjustment
        result[np.argmin(result)] = min_prob + adjustment
        min_val = min(result)
        max_val = max(result)
    return result

def randomly_distribute_N_balls_into_K_bins(N, K, *, max_balls_per_bin=None, min_balls_per_bin=None) -> np.ndarray:
    """Returns a uniformly random numpy integer array of length `N` that sums
    to `K`.

    Parameters
    ----------
    N : int
      Number of balls.
    K : int
      Number of bins.
    max_balls_per_bin : int
      Ensure that each bin contains at most `max_balls_per_bin` balls.
    min_balls_per_bin : int
      Ensure that each bin contains at least `min_balls_per_bin` balls.

    Returns
    -------
    int_array : np.array
      Length `N` array that sums to `K`.
    """
    if N == 0:
        return np.zeros(K, dtype=int)
    if max_balls_per_bin is None:
        max_balls_per_bin = N
    else:
        max_balls_per_bin = min(max_balls_per_bin, N)
    if min_balls_per_bin is None:
        min_balls_per_bin = 0
    else:
        min_balls_per_bin = min(min_balls_per_bin, N / K)
    if N / float(K) > max_balls_per_bin:
        N = max_balls_per_bin * K
    arr = np.round(generate_n_rand_probabilities_that_sum_to_m(n=K, m=1, max_prob=max_balls_per_bin / float(N), min_prob=min_balls_per_bin / float(N)) * N)
    while sum(arr) != N:
        while sum(arr) > N:
            arr[np.argmax(arr)] -= 1
        while sum(arr) < N:
            arr[np.argmin(arr)] += 1
    return arr.astype(int)
