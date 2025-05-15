def grid_to_graph(
    n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_matrix, dtype=int
):
    """Graph of the pixel-to-pixel connections.

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns
    -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import grid_to_graph
    >>> shape_img = (4, 4, 1)
    >>> mask = np.zeros(shape=shape_img, dtype=bool)
    >>> mask[[1, 2], [1, 2], :] = True
    >>> graph = grid_to_graph(*shape_img, mask=mask)
    >>> print(graph)
    <COOrdinate sparse matrix of dtype 'int64'
      with 2 stored elements and shape (2, 2)>
      Coords	Values
      (0, 0)    1
      (1, 1)    1
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as, dtype=dtype)