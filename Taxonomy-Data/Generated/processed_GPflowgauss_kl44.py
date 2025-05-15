import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    M, L = q_mu.shape
    
    if K is not None:
        if K_cholesky is not None:
            raise ValueError("Only one of K or K_cholesky should be provided.")
        if K.ndim == 2:
            K = np.broadcast_to(K, (L, M, M))
    elif K_cholesky is not None:
        if K_cholesky.ndim == 2:
            K_cholesky = np.broadcast_to(K_cholesky, (L, M, M))
        K = np.array([np.dot(K_cholesky[i], K_cholesky[i].T) for i in range(L)])
    else:
        K = np.eye(M)
        K = np.broadcast_to(K, (L, M, M))
    
    if q_sqrt.ndim == 2:
        q_sqrt = np.array([np.diag(q_sqrt[:, i]) for i in range(L)])
    
    kl_divergence = 0.0
    for i in range(L):
        q_cov = np.dot(q_sqrt[i], q_sqrt[i].T)
        K_inv = np.linalg.inv(K[i])
        
        trace_term = np.trace(np.dot(K_inv, q_cov))
        mean_term = np.dot(q_mu[:, i].T, np.dot(K_inv, q_mu[:, i]))
        log_det_q = np.sum(np.log(np.diag(q_sqrt[i])**2))
        log_det_K = np.linalg.slogdet(K[i])[1]
        
        kl_divergence += 0.5 * (trace_term + mean_term - M + log_det_K - log_det_q)
    
    return kl_divergence

