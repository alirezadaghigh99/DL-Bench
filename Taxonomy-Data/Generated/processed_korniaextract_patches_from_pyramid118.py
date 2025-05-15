import torch
import torch.nn.functional as F

def extract_patches_from_pyramid(img, laf, patch_size=32, normalize_lafs=False):
    B, CH, H, W = img.shape
    _, N, _, _ = laf.shape
    
    # Normalize LAFs if required
    if normalize_lafs:
        # Normalize LAFs to have unit determinant
        scales = torch.sqrt(torch.det(laf[:, :, :2, :2]))
        laf[:, :, :2, :2] /= scales.unsqueeze(-1).unsqueeze(-1)
    
    # Create image pyramid
    pyramid = [img]
    current_img = img
    while min(current_img.shape[-2:]) >= patch_size:
        current_img = F.interpolate(current_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        pyramid.append(current_img)
    
    # Determine the appropriate pyramid level for each LAF
    scales = torch.sqrt(torch.det(laf[:, :, :2, :2]))
    pyramid_levels = torch.clamp(torch.log2(scales).long(), 0, len(pyramid) - 1)
    
    # Extract patches
    patches = torch.zeros(B, N, CH, patch_size, patch_size, device=img.device)
    for b in range(B):
        for n in range(N):
            level = pyramid_levels[b, n].item()
            current_img = pyramid[level]
            scale_factor = 2 ** level
            
            # Transform LAF to the current pyramid level
            laf_scaled = laf[b, n].clone()
            laf_scaled[:, 2] /= scale_factor
            
            # Extract patch using affine grid
            grid = F.affine_grid(laf_scaled.unsqueeze(0), torch.Size((1, CH, patch_size, patch_size)), align_corners=False)
            patch = F.grid_sample(current_img[b:b+1], grid, mode='bilinear', align_corners=False)
            patches[b, n] = patch.squeeze(0)
    
    return patches

