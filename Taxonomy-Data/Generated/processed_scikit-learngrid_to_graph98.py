import numpy as np
from scipy.sparse import coo_matrix

def pixel_to_pixel_graph(n_x, n_y, n_z=1, mask=None, return_as=coo_matrix, dtype=int):
    # Total number of voxels
    total_voxels = n_x * n_y * n_z
    
    # Create a mask if none is provided
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)
    
    # Validate mask dimensions
    if mask.shape != (n_x, n_y, n_z):
        raise ValueError("Mask dimensions must match (n_x, n_y, n_z)")
    
    # Directions for 6-connectivity in 3D (up, down, left, right, front, back)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # Lists to store the row and column indices of the adjacency matrix
    rows = []
    cols = []
    
    # Iterate over each voxel
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                if not mask[x, y, z]:
                    continue
                
                # Current voxel index
                current_index = x * n_y * n_z + y * n_z + z
                
                # Check each direction for connectivity
                for dx, dy, dz in directions:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < n_x and 0 <= ny < n_y and 0 <= nz < n_z:
                        if mask[nx, ny, nz]:
                            neighbor_index = nx * n_y * n_z + ny * n_z + nz
                            rows.append(current_index)
                            cols.append(neighbor_index)
    
    # Create the adjacency matrix
    data = np.ones(len(rows), dtype=dtype)
    adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(total_voxels, total_voxels), dtype=dtype)
    
    # Return the adjacency matrix in the desired format
    if return_as == np.ndarray:
        return adjacency_matrix.toarray()
    else:
        return return_as(adjacency_matrix)

