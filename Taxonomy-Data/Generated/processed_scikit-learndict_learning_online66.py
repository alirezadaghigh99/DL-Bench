import numpy as np
from sklearn.utils import check_random_state, gen_batches, shuffle
from sklearn.linear_model import Lasso, lars_path
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.extmath import randomized_svd

def dict_learning_online(X, n_components=2, alpha=1, max_iter=100, return_code=True,
                         dict_init=None, callback=None, batch_size=256, verbose=False,
                         shuffle=True, n_jobs=None, method='lars', random_state=None,
                         positive_dict=False, positive_code=False, method_max_iter=1000,
                         tol=1e-3, max_no_improvement=10):
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)
    
    if dict_init is None:
        # Initialize dictionary using SVD
        _, S, Vt = randomized_svd(X, n_components)
        dictionary = Vt
    else:
        dictionary = dict_init

    if shuffle:
        X = shuffle(X, random_state=random_state)

    batches = gen_batches(n_samples, batch_size)
    n_batches = len(batches)

    best_cost = np.inf
    no_improvement = 0

    for iteration in range(max_iter):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iter}")

        for batch_idx, batch_slice in enumerate(batches):
            X_batch = X[batch_slice]

            # Sparse coding step
            if method == 'lars':
                _, _, coefs = lars_path(dictionary.T, X_batch.T, alpha=alpha, method='lasso')
                code = coefs.T
            elif method == 'cd':
                lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=method_max_iter,
                              positive=positive_code)
                code = np.array([lasso.fit(dictionary.T, x).coef_ for x in X_batch])

            # Dictionary update step
            for k in range(n_components):
                if np.any(code[:, k] != 0):
                    dictionary[k] = np.dot(code[:, k], X_batch) / np.dot(code[:, k], code[:, k])
                    if positive_dict:
                        dictionary[k] = np.maximum(dictionary[k], 0)
                    dictionary[k] /= np.linalg.norm(dictionary[k])

            # Calculate cost for early stopping
            reconstruction = np.dot(code, dictionary)
            cost = 0.5 * np.linalg.norm(X_batch - reconstruction, 'fro')**2 + alpha * np.sum(np.abs(code))
            if cost < best_cost - tol:
                best_cost = cost
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= max_no_improvement:
                if verbose:
                    print("Convergence reached: no improvement in cost function.")
                break

        if callback is not None:
            callback(locals())

    if return_code:
        return code, dictionary
    else:
        return dictionary

