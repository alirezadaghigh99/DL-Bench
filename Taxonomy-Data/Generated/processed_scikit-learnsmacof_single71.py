import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances

def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None,
                   normalized_stress=False):
    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    if init is None:
        X = random_state.rand(n_samples, n_components)
    else:
        X = init

    old_stress = None
    dissimilarities = np.asarray(dissimilarities)

    for it in range(max_iter):
        # Compute distance matrix
        distances = euclidean_distances(X)

        # Avoid division by zero
        np.fill_diagonal(distances, 1)

        # Compute stress
        if metric:
            disparities = dissimilarities
        else:
            disparities = np.copy(dissimilarities)
            disparities[disparities == 0] = np.nan
            disparities = np.nan_to_num(disparities)

        stress = np.sum((disparities - distances) ** 2)

        if normalized_stress and not metric:
            stress /= np.sum(disparities ** 2)

        if verbose:
            print(f"Iteration {it}: stress {stress}")

        if old_stress is not None and abs(old_stress - stress) < eps:
            if verbose:
                print(f"Converged at iteration {it} with stress {stress}")
            break

        old_stress = stress

        # Guttman transform
        ratio = disparities / distances
        B = -ratio
        np.fill_diagonal(B, ratio.sum(axis=1))
        X = np.dot(B, X) / n_samples

    return X, stress, it + 1