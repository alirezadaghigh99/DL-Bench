import numpy as np

def generate_noise_matrix_from_trace(K, trace, max_trace_prob, min_trace_prob, max_noise_rate, min_noise_rate,
                                     valid_noise_matrix=True, py=None, frac_zero_noise_rates=0.0, seed=None, max_iter=10000):
    if seed is not None:
        np.random.seed(seed)
    
    if valid_noise_matrix and (py is None or len(py) != K):
        raise ValueError("py must be provided and have length K when valid_noise_matrix is True.")
    
    for _ in range(max_iter):
        # Initialize the noise matrix
        noise_matrix = np.zeros((K, K))
        
        # Set diagonal elements to satisfy the trace condition
        diagonal_probs = np.random.uniform(min_trace_prob, max_trace_prob, K)
        diagonal_probs = diagonal_probs / np.sum(diagonal_probs) * trace
        
        if np.any(diagonal_probs > 1):
            continue
        
        np.fill_diagonal(noise_matrix, diagonal_probs)
        
        # Set off-diagonal elements
        for i in range(K):
            remaining_prob = 1 - noise_matrix[i, i]
            if remaining_prob < 0:
                break
            
            off_diagonal_probs = np.random.uniform(min_noise_rate, max_noise_rate, K - 1)
            off_diagonal_probs = off_diagonal_probs / np.sum(off_diagonal_probs) * remaining_prob
            
            if np.any(off_diagonal_probs > 1):
                break
            
            noise_matrix[i, np.arange(K) != i] = off_diagonal_probs
        
        # Check if the matrix is valid
        if valid_noise_matrix:
            valid = True
            for i in range(K):
                if not (py[i] * np.sum(noise_matrix[:, i]) < noise_matrix[i, i]):
                    valid = False
                    break
            if not valid:
                continue
        
        # Check the fraction of zero noise rates
        num_zero_noise_rates = np.sum(noise_matrix == 0) - K  # Exclude diagonal zeros
        total_off_diagonal = K * (K - 1)
        if num_zero_noise_rates / total_off_diagonal < frac_zero_noise_rates:
            continue
        
        return noise_matrix
    
    return None

