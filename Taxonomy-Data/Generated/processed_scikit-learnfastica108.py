import numpy as np
from scipy.linalg import svd, eigh
from sklearn.utils import check_random_state

def _logcosh(x, alpha=1.0):
    return np.tanh(alpha * x), alpha * (1 - np.tanh(alpha * x) ** 2)

def _exp(x):
    exp_x = np.exp(-x ** 2 / 2)
    return x * exp_x, (1 - x ** 2) * exp_x

def _cube(x):
    return x ** 3, 3 * x ** 2

def _whiten(X, n_components, whiten, whiten_solver):
    if whiten_solver == 'svd':
        U, S, Vt = svd(X, full_matrices=False)
        K = (Vt.T / S).T[:n_components]
        X_white = np.dot(K, X.T).T
    elif whiten_solver == 'eigh':
        cov = np.cov(X, rowvar=False)
        d, E = eigh(cov)
        D = np.diag(1.0 / np.sqrt(d))
        K = np.dot(E, D)[:n_components]
        X_white = np.dot(X - X.mean(axis=0), K.T)
    else:
        raise ValueError("Invalid whiten_solver option.")
    
    if whiten == "unit-variance":
        X_white /= np.std(X_white, axis=0)
    
    return X_white, K

def fastica(X, n_components=None, algorithm='parallel', whiten='unit-variance', fun='logcosh', fun_args=None,
            max_iter=200, tol=1e-04, w_init=None, whiten_solver='svd', random_state=None,
            return_X_mean=False, compute_sources=True, return_n_iter=False):
    
    if not isinstance(X, (np.ndarray, list)):
        raise TypeError("X should be array-like.")
    
    X = np.array(X)
    n_samples, n_features = X.shape
    
    if n_components is None:
        n_components = n_features
    
    if whiten not in ['unit-variance', 'arbitrary-variance', False]:
        raise ValueError("Invalid whiten option.")
    
    random_state = check_random_state(random_state)
    
    if fun == 'logcosh':
        g, g_prime = _logcosh
    elif fun == 'exp':
        g, g_prime = _exp
    elif fun == 'cube':
        g, g_prime = _cube
    elif callable(fun):
        g, g_prime = fun, fun_args.get('g_prime', None)
    else:
        raise ValueError("Invalid function option.")
    
    X_mean = X.mean(axis=0)
    X -= X_mean
    
    if whiten:
        X_white, K = _whiten(X, n_components, whiten, whiten_solver)
    else:
        X_white = X
        K = None
    
    W = random_state.normal(size=(n_components, n_components)) if w_init is None else w_init
    
    for i in range(max_iter):
        if algorithm == 'parallel':
            WX = np.dot(W, X_white.T)
            gwx, g_wx_prime = g(WX)
            W_new = np.dot(gwx, X_white) / n_samples - np.dot(np.diag(g_wx_prime.mean(axis=1)), W)
            W_new = np.dot(np.linalg.inv(np.sqrt(np.dot(W_new, W_new.T))), W_new)
        elif algorithm == 'deflation':
            for j in range(n_components):
                w = W[j, :]
                for _ in range(max_iter):
                    w_new = (X_white * g(np.dot(w, X_white.T))[0]).mean(axis=1) - g_prime(np.dot(w, X_white.T)).mean() * w
                    w_new /= np.linalg.norm(w_new)
                    if np.abs(np.abs((w_new * w).sum()) - 1) < tol:
                        break
                    w = w_new
                W[j, :] = w
        else:
            raise ValueError("Invalid algorithm option.")
        
        if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < tol:
            break
        W = W_new
    
    if compute_sources:
        S = np.dot(W, X_white.T).T
    else:
        S = None
    
    result = [K, W, S]
    if return_X_mean:
        result.append(X_mean)
    if return_n_iter:
        result.append(i + 1)
    
    return result