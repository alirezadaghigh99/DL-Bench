import numpy as np
from scipy.linalg import qr

def cluster_qr(vectors):
    """
    Finds the discrete partition closest to the eigenvector embedding.

    Parameters:
    vectors (array-like): An array with shape (n_samples, n_clusters) representing the embedding space of the samples.

    Returns:
    labels (array): An array of integers with shape (n_samples,) representing the cluster labels of the vectors.
    """
    # Ensure the input is a numpy array
    vectors = np.asarray(vectors)
    
    # Perform QR decomposition on the transpose of the vectors
    Q, R = qr(vectors.T, mode='economic')
    
    # The cluster labels are determined by the index of the maximum value in each row of Q
    labels = np.argmax(Q, axis=1)
    
    return labels

