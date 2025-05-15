import torch
import torch.nn.functional as F

def yuv_to_rgb(yuv):
    # YUV to RGB conversion matrix
    # Assuming YUV is in the range Y: [0, 1], U, V: [-0.5, 0.5]
    # RGB will be in the range [0, 1]
    y, u, v = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]
    
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    
    rgb = torch.stack((r, g, b), dim=1)
    return rgb.clamp(0, 1)

def yuv420_to_rgb(imagey, imageuv):
    # Check input types and shapes
    assert isinstance(imagey, torch.Tensor), "imagey must be a torch Tensor"
    assert isinstance(imageuv, torch.Tensor), "imageuv must be a torch Tensor"
    assert imagey.shape[1] == 1, "imagey must have shape (*, 1, H, W)"
    assert imageuv.shape[1] == 2, "imageuv must have shape (*, 2, H/2, W/2)"
    
    batch_size, _, H, W = imagey.shape
    _, _, H_uv, W_uv = imageuv.shape
    
    assert H == 2 * H_uv and W == 2 * W_uv, "imageuv dimensions must be half of imagey dimensions"
    
    # Upsample UV to match Y dimensions
    imageuv_upsampled = F.interpolate(imageuv, size=(H, W), mode='bilinear', align_corners=False)
    
    # Concatenate Y and upsampled UV to form YUV444
    yuv444 = torch.cat((imagey, imageuv_upsampled), dim=1)
    
    # Convert YUV444 to RGB
    rgb = yuv_to_rgb(yuv444)
    
    return rgb

