import torch
import torch.nn.functional as F

def vert_align(feats, verts, return_packed=False, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    # Validate input shapes and attributes
    if not isinstance(feats, (list, torch.Tensor)):
        raise ValueError("feats must be a tensor or a list of tensors.")
    
    if isinstance(feats, torch.Tensor):
        feats = [feats]
    
    for f in feats:
        if len(f.shape) != 4:
            raise ValueError("Each feature map in feats must have shape (N, C, H, W).")
    
    if hasattr(verts, 'verts_padded'):
        verts = verts.verts_padded()
    elif hasattr(verts, 'points_padded'):
        verts = verts.points_padded()
    elif not isinstance(verts, torch.Tensor) or len(verts.shape) != 3 or verts.shape[2] != 3:
        raise ValueError("verts must be a tensor of shape (N, V, 3) or have verts_padded/points_padded attributes.")
    
    N = verts.shape[0]
    if any(f.shape[0] != N for f in feats):
        raise ValueError("Batch dimension of feats and verts must match.")
    
    # Prepare to sample features
    sampled_features = []
    
    for f in feats:
        N, C, H, W = f.shape
        
        # Normalize vertex positions to the feature map dimensions
        grid = verts[..., :2].clone()
        grid = grid.view(N, -1, 1, 2)  # Reshape for grid_sample
        grid[..., 0] = (grid[..., 0] + 1) * 0.5 * (W - 1) / W
        grid[..., 1] = (grid[..., 1] + 1) * 0.5 * (H - 1) / H
        
        # Sample features using grid_sample
        sampled = F.grid_sample(f, grid, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
        sampled = sampled.view(N, C, -1).permute(0, 2, 1)  # Reshape to (N, V, C)
        
        sampled_features.append(sampled)
    
    # Concatenate features from all feature maps
    feats_sampled = torch.cat(sampled_features, dim=2)  # Concatenate along the channel dimension
    
    if return_packed:
        feats_sampled = feats_sampled.view(-1, feats_sampled.shape[-1])  # Transform to packed representation
    
    return feats_sampled