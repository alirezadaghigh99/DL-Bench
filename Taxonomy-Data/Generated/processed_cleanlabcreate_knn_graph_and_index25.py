import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def create_knn_graph_and_index(features, n_neighbors=5, metric='euclidean', correct_duplicates=False, **kwargs):
    """
    Create a KNN graph and a KNN search index from input features.

    Parameters:
    - features: np.ndarray, shape (n_samples, n_features)
        The input feature array.
    - n_neighbors: int, optional (default=5)
        Number of nearest neighbors to use.
    - metric: str, optional (default='euclidean')
        The distance metric to use for the tree.
    - correct_duplicates: bool, optional (default=False)
        Whether to correct for exact duplicates in the feature array.
    - **kwargs: additional keyword arguments for the NearestNeighbors constructor.

    Returns:
    - adjacency_matrix: scipy.sparse.csr_matrix
        Sparse, weighted adjacency matrix representing the KNN graph.
    - knn_search: NearestNeighbors
        Fitted KNN search object.
    """
    # Handle exact duplicates if required
    if correct_duplicates:
        features = np.unique(features, axis=0)

    # Initialize the NearestNeighbors object
    knn_search = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    
    # Fit the model
    knn_search.fit(features)
    
    # Find the k-neighbors for each sample
    distances, indices = knn_search.kneighbors(features)
    
    # Create a sparse adjacency matrix
    n_samples = features.shape[0]
    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = indices.flatten()
    data = distances.flatten()
    
    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    
    return adjacency_matrix, knn_search

