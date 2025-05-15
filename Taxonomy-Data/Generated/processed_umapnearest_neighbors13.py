import numpy as np
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

def nearest_neighbors(X, n_neighbors=5, metric='euclidean', metric_params=None, 
                      use_angular_rp_trees=False, random_state=None, 
                      low_memory=False, verbose=False):
    """
    Compute the n_neighbors nearest points for each data point in X under a specified metric.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        The input data.
    - n_neighbors: int, default=5
        Number of nearest neighbors to compute.
    - metric: str or callable, default='euclidean'
        The distance metric to use.
    - metric_params: dict, default=None
        Additional keyword arguments for the metric function.
    - use_angular_rp_trees: bool, default=False
        Whether to use angular random projection trees for approximation.
    - random_state: int or None, default=None
        Random state for approximate computations.
    - low_memory: bool, default=False
        If True, use a memory-efficient algorithm.
    - verbose: bool, default=False
        If True, print status data during computation.

    Returns:
    - knn_indices: array, shape (n_samples, n_neighbors)
        Indices of the n_neighbors closest points in the dataset.
    - knn_dists: array, shape (n_samples, n_neighbors)
        Distances to the n_neighbors closest points in the dataset.
    - rp_forest: list or None
        List of trees representing the random projection forest used for searching, or None if not used.
    """
    if use_angular_rp_trees:
        if verbose:
            print("Using Annoy for approximate nearest neighbors with angular distance.")
        
        # Initialize Annoy index
        n_features = X.shape[1]
        annoy_index = AnnoyIndex(n_features, 'angular')
        
        # Add items to the Annoy index
        for i, x in enumerate(X):
            annoy_index.add_item(i, x)
        
        # Build the index
        annoy_index.build(10, n_jobs=-1)  # 10 trees, using all available cores
        
        # Query the index
        knn_indices = []
        knn_dists = []
        for i in range(X.shape[0]):
            indices, dists = annoy_index.get_nns_by_item(i, n_neighbors, include_distances=True)
            knn_indices.append(indices)
            knn_dists.append(dists)
        
        knn_indices = np.array(knn_indices)
        knn_dists = np.array(knn_dists)
        rp_forest = [annoy_index]
    else:
        if verbose:
            print("Using scikit-learn for exact nearest neighbors.")
        
        # Initialize NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=metric_params, 
                              algorithm='auto', n_jobs=-1)
        nn.fit(X)
        
        # Find nearest neighbors
        knn_dists, knn_indices = nn.kneighbors(X)
        rp_forest = None
    
    return knn_indices, knn_dists, rp_forest

