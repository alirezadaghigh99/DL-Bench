import torch
import torch.nn.functional as F

from pytorch3d.structures.utils import padded_to_packed


def vert_align(feats, verts, return_packed: bool=False, interp_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=True) -> torch.Tensor:
    """Create a Python function vert_align that performs "vertex alignment" or "perceptual feature pooling," which samples vertex features from a feature map based on vertex positions. The function takes the following parameters:

feats: A tensor of shape (N, C, H, W) representing image features from which to sample, or a list of such tensors, each with potentially different C, H, or W dimensions.
verts: A tensor of shape (N, V, 3) representing the (x, y, z) vertex positions, or an object with verts_padded or points_padded attributes. The (x, y) coordinates should be normalized such that (-1, -1) corresponds to the top-left and (1, 1) to the bottom-right of the feature map.
return_packed: A boolean flag indicating whether to return packed features. Defaults to False.
interp_mode: A string specifying the interpolation mode ('bilinear' or 'nearest'). Defaults to 'bilinear'.
padding_mode: A string specifying how to handle vertices outside the [-1, 1] range ('zeros', 'reflection', or 'border'). Defaults to 'zeros'.
align_corners: A boolean indicating whether to align corners geometrically. If True, extrema refer to the center points of corner pixels; if False, they refer to the corner points of the input's corner pixels. Defaults to True.
Returns:
feats_sampled: A tensor of shape (N, V, C) giving sampled features for each vertex. If feats is a list, the function returns concatenated features in shape (N, V, sum(C_n)) where C_n = feats[n].shape[1]. If return_packed = True, the features are transformed to a packed representation of shape (sum(V), C).
Error Handling:
Raise a ValueError if verts does not have the expected shape or attributes.
Raise a ValueError if feats does not have the expected shape (N, C, H, W) or if the batch dimensions of feats and verts do not match."""
    if torch.is_tensor(verts):
        if verts.dim() != 3:
            raise ValueError("verts tensor should be 3 dimensional")
        grid = verts
    elif hasattr(verts, "verts_padded"):
        grid = verts.verts_padded()
    elif hasattr(verts, "points_padded"):
        grid = verts.points_padded()
    else:
        raise ValueError(
            "verts must be a tensor or have a "
            "'verts_padded' or 'points_padded' attribute."
        )

    grid = grid[:, None, :, :2]  # (N, 1, V, 2)

    if torch.is_tensor(feats):
        feats = [feats]
    for feat in feats:
        if feat.dim() != 4:
            raise ValueError("feats must have shape (N, C, H, W)")
        if grid.shape[0] != feat.shape[0]:
            raise ValueError("inconsistent batch dimension")

    feats_sampled = []
    for feat in feats:
        feat_sampled = F.grid_sample(
            feat,
            grid,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feats_sampled.append(feat_sampled)
    feats_sampled = torch.cat(feats_sampled, dim=2)  # (N, V, sum(C))

    if return_packed:
        if not torch.is_tensor(verts):
            if hasattr(verts, "verts_padded"):
                num_verts = verts.num_verts_per_mesh()
            else:
                num_verts = verts.num_points_per_cloud()
            split_size = num_verts.tolist()
            feats_sampled = padded_to_packed(feats_sampled, split_size=split_size)
        else:
            feats_sampled = feats_sampled.reshape(-1, feats_sampled.shape[-1])

    return feats_sampled
