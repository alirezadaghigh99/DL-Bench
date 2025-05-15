import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import cubify

def cubify(voxels, thresh, feats=None, device='cpu', align='center'):
    """
    Convert a voxel grid to a mesh representation.

    Args:
        voxels (torch.Tensor): A tensor of shape (N, D, H, W) containing occupancy probabilities.
        thresh (float): A scalar threshold for occupancy.
        feats (torch.Tensor, optional): A tensor of shape (N, K, D, H, W) containing color information.
        device (str): The device of the output meshes.
        align (str): The alignment of the mesh vertices and grid locations. Can be "topleft", "corner", or "center".

    Returns:
        Meshes: A Meshes object containing the cubified meshes.
    """
    # Ensure the voxels are on the correct device
    voxels = voxels.to(device)

    # Binarize the voxel grid based on the threshold
    binary_voxels = (voxels > thresh).float()

    # Use PyTorch3D's cubify function to convert the voxel grid to a mesh
    # The cubify function expects a binary occupancy grid
    meshes = cubify(binary_voxels, thresh=0.5, align=align)

    # If feats are provided, attach them to the meshes
    if feats is not None:
        feats = feats.to(device)
        # Flatten the features to match the vertices
        # Assuming feats are in the same order as the voxels
        # This part may need adjustment based on how features are intended to be used
        vertex_feats = feats.permute(0, 2, 3, 4, 1).reshape(-1, feats.size(1))
        meshes.textures = vertex_feats

    return meshes

