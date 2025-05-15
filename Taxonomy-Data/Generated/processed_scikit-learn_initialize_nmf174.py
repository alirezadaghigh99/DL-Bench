import numpy as np
from sklearn.utils import check_random_state
from sklearn.decomposition import TruncatedSVD

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)
    
    if init == 'random':
        W = rng.rand(n_samples, n_components)
        H = rng.rand(n_components, n_features)
        W[W < eps] = 0
        H[H < eps] = 0
        return W, H
    
    elif init in {'nndsvd', 'nndsvda', 'nndsvdar'}:
        # Compute SVD of X
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        U = svd.fit_transform(X)
        S = svd.singular_values_
        V = svd.components_
        
        # Initialize W and H
        W = np.zeros((n_samples, n_components))
        H = np.zeros((n_components, n_features))
        
        # The first singular triplet
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
        
        # The other singular triplets
        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]
            xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
            yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
            xpnorm, xnnorm = np.linalg.norm(xp), np.linalg.norm(xn)
            ypnorm, ynnorm = np.linalg.norm(yp), np.linalg.norm(yn)
            m = xpnorm * ypnorm
            n = xnnorm * ynnorm
            if m > n:
                W[:, j] = np.sqrt(S[j] * m) * xp / xpnorm
                H[j, :] = np.sqrt(S[j] * m) * yp / ypnorm
            else:
                W[:, j] = np.sqrt(S[j] * n) * xn / xnnorm
                H[j, :] = np.sqrt(S[j] * n) * yn / ynnorm
        
        if init == 'nndsvda':
            W[W < eps] = eps
            H[H < eps] = eps
        elif init == 'nndsvdar':
            W[W < eps] = rng.rand(np.sum(W < eps)) * eps
            H[H < eps] = rng.rand(np.sum(H < eps)) * eps
        
        return W, H
    
    else:
        raise ValueError("Invalid init parameter. Choose from {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}.")

