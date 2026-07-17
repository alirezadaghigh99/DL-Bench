from typing import Optional
import torch
import torch.nn.functional as F
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.structures import Meshes

def unravel_index(idx, dims) -> torch.Tensor:
    """
    Equivalent to np.unravel_index
    Args:
      idx: A LongTensor whose elements are indices into the
          flattened version of an array of dimensions dims.
      dims: The shape of the array to be indexed.
    Implemented only for dims=(N, H, W, D)
    """
    if len(dims) != 4:
        raise ValueError('Expects a 4-element list.')
    (N, H, W, D) = dims
    n = idx // (H * W * D)
    h = (idx - n * H * W * D) // (W * D)
    w = (idx - n * H * W * D - h * W * D) // D
    d = idx - n * H * W * D - h * W * D - w * D
    return torch.stack((n, h, w, d), dim=1)

def ravel_index(idx, dims) -> torch.Tensor:
    """
    Computes the linear index in an array of shape dims.
    It performs the reverse functionality of unravel_index
    Args:
      idx: A LongTensor of shape (N, 3). Each row corresponds to indices into an
          array of dimensions dims.
      dims: The shape of the array to be indexed.
    Implemented only for dims=(H, W, D)
    """
    if len(dims) != 3:
        raise ValueError('Expects a 3-element list')
    if idx.shape[1] != 3:
        raise ValueError('Expects an index tensor of shape Nx3')
    (H, W, D) = dims
    linind = idx[:, 0] * W * D + idx[:, 1] * D + idx[:, 2]
    return linind

@torch.no_grad()
def cubify(voxels: torch.Tensor, thresh: float, *, feats: Optional[torch.Tensor]=None, device=None, align: str='topleft') -> Meshes:
    """Converts a voxel grid into a mesh by replacing every occupied voxel with a
    cuboid of 12 triangular faces and 8 vertices. Vertices shared between
    neighboring cubes are merged, and faces shared between two occupied
    (and therefore interior) voxels are removed, so the result is a
    watertight mesh of the voxelized shape.

    Args:
        voxels: A Tensor of shape (N, D, H, W) containing occupancy
            probabilities.
        thresh: A scalar threshold. A voxel is occupied if its value is
            strictly greater than thresh.
        feats: An optional Tensor of shape (N, K, D, H, W) containing a
            K-dimensional color/feature vector per voxel. If given, the
            returned Meshes will carry a TexturesVertex where each vertex's
            feature is the average of the features of the occupied voxels
            that share that corner.
        device: The device of the output meshes. Defaults to voxels.device.
        align: One of "topleft", "corner", "center", defining how the cube
            corners (integer grid points 0..D, 0..H, 0..W) are mapped to
            vertex positions along each axis of extent S (S = D, H or W):

            - "topleft": no rescaling; corner p maps to p. The mesh spans
              [0, D] x [0, H] x [0, W].
            - "corner": corner p maps to 2 * p / S - 1, so the outer
              corners of the whole grid land exactly on -1 and 1.
            - "center": corner p maps to (2 * p - S) / (S - 1), so the
              *centers* of the outermost voxels land on -1 and 1 (matching
              the extreme corners falling slightly outside [-1, 1]).

    Returns:
        meshes: A Meshes object with N meshes.
    """
    if align not in ("topleft", "corner", "center"):
        raise ValueError('Align mode must be one of ("topleft", "corner", "center").')
    if voxels.dim() != 4:
        raise ValueError("voxels must have shape (N, D, H, W).")

    if device is None:
        device = voxels.device

    N, D, H, W = voxels.shape
    voxels = voxels.to(device)
    occ = voxels > thresh  # (N, D, H, W)

    if feats is not None:
        feats = feats.to(device)
        if feats.shape[0] != N or tuple(feats.shape[2:]) != (D, H, W):
            raise ValueError(
                "feats must have shape (N, K, D, H, W) matching voxels."
            )

    if N == 0:
        return Meshes(verts=[], faces=[])

    # 8 corners of a unit cube, indexed as dD * 4 + dH * 2 + dW.
    cube_corners = torch.tensor(
        [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ],
        dtype=torch.int64,
        device=device,
    )  # (8, 3), columns are (d, h, w) offsets

    # The 6 faces of a cube (2 outward-winding triangles each), as local
    # corner indices into cube_corners. Order: -D, +D, -H, +H, -W, +W.
    cube_faces = torch.tensor(
        [
            [0, 2, 3], [0, 3, 1],
            [4, 5, 7], [4, 7, 6],
            [0, 1, 5], [0, 5, 4],
            [2, 6, 7], [2, 7, 3],
            [0, 4, 6], [0, 6, 2],
            [1, 3, 7], [1, 7, 5],
        ],
        dtype=torch.int64,
        device=device,
    )  # (12, 3)
    face_of_tri = torch.arange(6, device=device).repeat_interleave(2)  # (12,)

    # A face between a voxel and a neighbor is kept only if the voxel is
    # occupied and that neighbor (possibly out of grid bounds) is not.
    padded = F.pad(occ.to(torch.uint8), (1, 1, 1, 1, 1, 1), mode="constant", value=0).bool()
    faces_exposed = torch.stack(
        [
            occ & ~padded[:, 0:D, 1:H + 1, 1:W + 1],  # -D neighbor
            occ & ~padded[:, 2:D + 2, 1:H + 1, 1:W + 1],  # +D neighbor
            occ & ~padded[:, 1:D + 1, 0:H, 1:W + 1],  # -H neighbor
            occ & ~padded[:, 1:D + 1, 2:H + 2, 1:W + 1],  # +H neighbor
            occ & ~padded[:, 1:D + 1, 1:H + 1, 0:W],  # -W neighbor
            occ & ~padded[:, 1:D + 1, 1:H + 1, 2:W + 2],  # +W neighbor
        ],
        dim=-1,
    )  # (N, D, H, W, 6)

    grid_strides = torch.tensor([(H + 1) * (W + 1), W + 1, 1], dtype=torch.int64, device=device)

    def align_axis(p: torch.Tensor, size: int) -> torch.Tensor:
        p = p.to(voxels.dtype)
        if align == "topleft":
            return p
        if align == "corner":
            return 2.0 * p / size - 1.0
        denom = max(size - 1, 1)
        return (2.0 * p - size) / denom

    verts_list = []
    faces_list = []
    feats_list = [] if feats is not None else None

    for n in range(N):
        occ_idx = occ[n].nonzero(as_tuple=False)  # (M, 3): (d, h, w) of occupied voxels
        empty_verts = voxels.new_zeros((0, 3))
        empty_faces = torch.zeros((0, 3), dtype=torch.int64, device=device)

        if occ_idx.numel() == 0:
            verts_list.append(empty_verts)
            faces_list.append(empty_faces)
            if feats is not None:
                feats_list.append(feats.new_zeros((0, feats.shape[1])))
            continue

        vis = faces_exposed[n][occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]]  # (M, 6)

        corners = occ_idx[:, None, :] + cube_corners[None, :, :]  # (M, 8, 3)
        corner_ids = (corners * grid_strides).sum(dim=-1)  # (M, 8), unique per grid corner

        tris = corner_ids[:, cube_faces]  # (M, 12, 3)
        tri_vis = vis[:, face_of_tri]  # (M, 12)

        kept = tris.reshape(-1, 3)[tri_vis.reshape(-1)]  # (F, 3) global corner ids per kept tri

        if kept.numel() == 0:
            verts_list.append(empty_verts)
            faces_list.append(empty_faces)
            if feats is not None:
                feats_list.append(feats.new_zeros((0, feats.shape[1])))
            continue

        unique_ids, inverse = torch.unique(kept.reshape(-1), sorted=True, return_inverse=True)
        local_faces = inverse.reshape(-1, 3)

        i = unique_ids // ((H + 1) * (W + 1))
        rem = unique_ids % ((H + 1) * (W + 1))
        j = rem // (W + 1)
        k = rem % (W + 1)

        verts = torch.stack(
            [align_axis(k, W), align_axis(j, H), align_axis(i, D)], dim=-1
        )  # (V, 3), columns are (x, y, z)

        verts_list.append(verts)
        faces_list.append(local_faces)

        if feats is not None:
            K = feats.shape[1]
            voxel_feats = feats[n, :, occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]].transpose(0, 1)  # (M, K)
            flat_ids = corner_ids.reshape(-1)  # (M * 8,)
            flat_feats = voxel_feats.unsqueeze(1).expand(-1, 8, -1).reshape(-1, K)

            pos = torch.searchsorted(unique_ids, flat_ids).clamp(max=unique_ids.shape[0] - 1)
            mask = unique_ids[pos] == flat_ids
            local_idx = pos[mask]

            sum_feats = feats.new_zeros((unique_ids.shape[0], K))
            count = feats.new_zeros((unique_ids.shape[0],))
            sum_feats.index_add_(0, local_idx, flat_feats[mask])
            count.index_add_(0, local_idx, torch.ones_like(local_idx, dtype=feats.dtype))
            feats_list.append(sum_feats / count.clamp(min=1).unsqueeze(1))

    textures = None
    if feats is not None:
        from pytorch3d.renderer.mesh.textures import TexturesVertex

        textures = TexturesVertex(verts_features=feats_list)

    return Meshes(verts=verts_list, faces=faces_list, textures=textures)
