import numpy as np
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.sparse import csgraph

def tswspectral_layout(data=None, graph=None, dim=2, random_state=None, metric='euclidean', 
                       metric_kwds=None, method='truncated_svd', tol=1e-5, maxiter=1000):
    """
    Compute the spectral embedding of a graph using a truncated SVD-based approach.

    Parameters:
    - data: Optional, data to construct the graph if graph is not provided.
    - graph: NetworkX graph object. If not provided, data must be used to construct the graph.
    - dim: Number of dimensions for the embedding.
    - random_state: Seed for random number generator.
    - metric: Distance metric to use (not used in this function, but kept for compatibility).
    - metric_kwds: Additional keyword arguments for the metric (not used in this function).
    - method: Method to use for SVD ('truncated_svd' is the default and only option here).
    - tol: Tolerance for convergence.
    - maxiter: Maximum number of iterations for the SVD solver.

    Returns:
    - embedding: A numpy array of shape (n_nodes, dim) representing the spectral embedding.
    """
    if graph is None:
        if data is None:
            raise ValueError("Either 'graph' or 'data' must be provided.")
        # Construct the graph from data if graph is not provided
        graph = nx.from_numpy_matrix(data) if isinstance(data, np.ndarray) else nx.Graph(data)

    # Compute the normalized Laplacian matrix
    laplacian = csgraph.laplacian(nx.to_scipy_sparse_matrix(graph), normed=True)

    # Use truncated SVD to find the eigenvectors
    # We need the smallest `dim + 1` eigenvectors, but skip the first one (corresponding to eigenvalue 0)
    u, s, vt = svds(laplacian, k=dim + 1, tol=tol, maxiter=maxiter, which='SM', return_singular_vectors='u')

    # The embedding is given by the eigenvectors corresponding to the smallest non-zero eigenvalues
    embedding = u[:, 1:dim + 1]

    return embedding

