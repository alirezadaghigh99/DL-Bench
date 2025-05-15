import numpy as np

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None,
                     weights=None, batch_reduction='mean', point_reduction='mean',
                     norm=2, single_directional=False, abs_cosine=False):
    """
    Calculate the Chamfer distance between two point clouds x and y.

    Parameters:
    - x, y: np.ndarray, shape (B, N, D) and (B, M, D) respectively, where B is the batch size,
      N and M are the number of points in each point cloud, and D is the dimensionality.
    - x_lengths, y_lengths: Optional, lengths of each point cloud in the batch.
    - x_normals, y_normals: Optional, normals of each point in the point clouds.
    - weights: Optional, weights for each point in the point clouds.
    - batch_reduction: str, reduction method over the batch ('mean', 'sum', 'none').
    - point_reduction: str, reduction method over the points ('mean', 'sum', 'none').
    - norm: int, norm degree for distance calculation (1 for L1, 2 for L2).
    - single_directional: bool, if True, compute only one directional distance.
    - abs_cosine: bool, if True, compute absolute cosine distance for normals.

    Returns:
    - Tuple of reduced distance and reduced cosine distance of normals.
    """
    def pairwise_distances(a, b, norm=2):
        if norm == 1:
            return np.sum(np.abs(a[:, :, None, :] - b[:, None, :, :]), axis=-1)
        elif norm == 2:
            return np.sqrt(np.sum((a[:, :, None, :] - b[:, None, :, :]) ** 2, axis=-1))
        else:
            raise ValueError("Unsupported norm type. Use 1 or 2.")

    def reduce_distances(distances, reduction='mean'):
        if reduction == 'mean':
            return np.mean(distances, axis=-1)
        elif reduction == 'sum':
            return np.sum(distances, axis=-1)
        elif reduction == 'none':
            return distances
        else:
            raise ValueError("Unsupported reduction type. Use 'mean', 'sum', or 'none'.")

    # Compute pairwise distances
    dist_x_to_y = pairwise_distances(x, y, norm=norm)
    dist_y_to_x = pairwise_distances(y, x, norm=norm)

    # Reduce distances
    min_dist_x_to_y = np.min(dist_x_to_y, axis=-1)
    min_dist_y_to_x = np.min(dist_y_to_x, axis=-1)

    if not single_directional:
        chamfer_dist = reduce_distances(min_dist_x_to_y, point_reduction) + \
                       reduce_distances(min_dist_y_to_x, point_reduction)
    else:
        chamfer_dist = reduce_distances(min_dist_x_to_y, point_reduction)

    # Handle batch reduction
    if batch_reduction == 'mean':
        chamfer_dist = np.mean(chamfer_dist, axis=0)
    elif batch_reduction == 'sum':
        chamfer_dist = np.sum(chamfer_dist, axis=0)

    # Compute cosine distance of normals if provided
    if x_normals is not None and y_normals is not None:
        dot_product = np.sum(x_normals[:, :, None, :] * y_normals[:, None, :, :], axis=-1)
        if abs_cosine:
            cosine_dist = 1 - np.abs(dot_product)
        else:
            cosine_dist = 1 - dot_product

        min_cosine_dist_x_to_y = np.min(cosine_dist, axis=-1)
        min_cosine_dist_y_to_x = np.min(cosine_dist, axis=-1)

        if not single_directional:
            cosine_dist = reduce_distances(min_cosine_dist_x_to_y, point_reduction) + \
                          reduce_distances(min_cosine_dist_y_to_x, point_reduction)
        else:
            cosine_dist = reduce_distances(min_cosine_dist_x_to_y, point_reduction)

        # Handle batch reduction for normals
        if batch_reduction == 'mean':
            cosine_dist = np.mean(cosine_dist, axis=0)
        elif batch_reduction == 'sum':
            cosine_dist = np.sum(cosine_dist, axis=0)
    else:
        cosine_dist = None

    return chamfer_dist, cosine_dist