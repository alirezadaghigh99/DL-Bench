from typing import Tuple
import torch

def laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]
    (e0, e1) = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)
    idx10 = torch.stack([e1, e0], dim=1)
    idx = torch.cat([idx01, idx10], dim=0).t()
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))
    return L

def cot_laplacian(verts: torch.Tensor, faces: torch.Tensor, eps: float=1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        faces: tensor of shape (F, 3) containing the vertex indices of each face
    Returns:
        2-element tuple containing
        - **L**: Sparse FloatTensor of shape (V, V) for the Laplacian matrix.
          Here, L[i, j] = cot a_ij + cot b_ij, where a_ij and b_ij are the
          angles at the two vertices opposite to the edge (i, j).
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of the
          sum of the areas of triangles containing each vertex.
    """
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (F,)
    # A is the side opposite v0, B is opposite v1, and C is opposite v2
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (Heron's formula), shape is (F,)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

    # Compute cotangents of angles, of shape (F, 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of the areas of triangles containing it.
    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area, area, area], dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(V, 1)

    return L, inv_areas

def norm_laplacian(verts: torch.Tensor, edges: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """
    Norm laplacian computes a variant of the laplacian matrix which weights each
    affinity with the normalized distance of the neighboring nodes.
    More concretely,
    L[i, j] = 1. / wij where wij = ||vi - vj|| if (vi, vj) are neighboring nodes

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    edge_verts = verts[edges]
    (v0, v1) = (edge_verts[:, 0], edge_verts[:, 1])
    w01 = 1.0 / ((v0 - v1).norm(dim=1) + eps)
    e01 = edges.t()
    V = verts.shape[0]
    L = torch.sparse.FloatTensor(e01, w01, (V, V))
    L = L + L.t()
    return L
