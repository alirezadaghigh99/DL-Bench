import numpy as np
from sklearn.feature_extraction.image import img_to_graph as sklearn_img_to_graph
from scipy import sparse

def img_to_graph(img, mask=None, return_as=sparse.csr_matrix, dtype=np.float64):
    """
    Generate a graph of pixel-to-pixel gradient connections from a 2D or 3D image.

    Parameters:
    - img: array-like of shape (height, width) or (height, width, channel)
        The input image.
    - mask: array-like of shape (height, width), optional
        An optional mask. If provided, only the pixels selected by the mask are used.
    - return_as: class, optional
        The class to build the adjacency matrix. Default is scipy.sparse.csr_matrix.
    - dtype: data type, optional
        The data type of the returned sparse matrix. Default is np.float64.

    Returns:
    - adjacency_matrix: ndarray or sparse matrix
        The computed adjacency matrix as either an ndarray or a sparse matrix class.
    """
    # Ensure the image is a numpy array
    img = np.asarray(img)

    # If the image is 3D, reshape it to 2D by combining the color channels
    if img.ndim == 3:
        img = img.reshape(-1, img.shape[-1])

    # Use sklearn's img_to_graph to generate the adjacency matrix
    adjacency_matrix = sklearn_img_to_graph(img, mask=mask, return_as=return_as, dtype=dtype)

    return adjacency_matrix

