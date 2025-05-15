from sklearn.decomposition import dict_learning
import numpy as np

def dict_learning_custom(X, n_components, alpha, max_iter=100, tol=1e-8, method='lars',
                         n_jobs=None, dict_init=None, code_init=None, callback=None,
                         verbose=False, random_state=None, return_n_iter=False,
                         positive_dict=False, positive_code=False, method_max_iter=1000):
    """
    Solves a dictionary learning matrix factorization problem.

    Parameters:
    - X: Data matrix of shape (n_samples, n_features)
    - n_components: Number of dictionary atoms to extract
    - alpha: Sparsity controlling parameter
    - max_iter: Maximum number of iterations to perform
    - tol: Tolerance for the stopping condition
    - method: Method used for solving the problem
    - n_jobs: Number of parallel jobs to run
    - dict_init: Initial value for the dictionary for warm restart scenarios
    - code_init: Initial value for the sparse code for warm restart scenarios
    - callback: Callable that gets invoked every five iterations
    - verbose: Verbosity of the procedure
    - random_state: Used for randomly initializing the dictionary
    - return_n_iter: Whether or not to return the number of iterations
    - positive_dict: Whether to enforce positivity when finding the dictionary
    - positive_code: Whether to enforce positivity when finding the code
    - method_max_iter: Maximum number of iterations to perform

    Returns:
    - code: Sparse code factor in the matrix factorization
    - dictionary: Dictionary factor in the matrix factorization
    - errors: Vector of errors at each iteration
    """
    # Perform dictionary learning
    code, dictionary, errors = dict_learning(
        X, n_components=n_components, alpha=alpha, max_iter=max_iter, tol=tol,
        method=method, n_jobs=n_jobs, dict_init=dict_init, code_init=code_init,
        callback=callback, verbose=verbose, random_state=random_state,
        return_n_iter=return_n_iter, positive_dict=positive_dict,
        positive_code=positive_code, method_max_iter=method_max_iter
    )
    
    return code, dictionary, errors

