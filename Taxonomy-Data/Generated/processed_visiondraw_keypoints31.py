import torch
from PIL import Image, ImageDraw
import numpy as np

def draw_keypoints(image, keypoints, connectivity=None, colors='red', radius=3, width=2, visibility=None):
    # Validate image tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("The image must be a tensor.")
    
    if image.dtype not in [torch.uint8, torch.float]:
        raise ValueError("The image dtype must be uint8 or float.")
    
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("The image must have shape (3, H, W).")
    
    # Convert image to uint8 if it's float
    if image.dtype == torch.float:
        image = (image * 255).clamp(0, 255).byte()
    
    # Convert image tensor to PIL Image
    image_np = image.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_image)
    
    # Validate keypoints tensor
    if not isinstance(keypoints, torch.Tensor) or keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError("Keypoints must be a tensor of shape (num_instances, K, 2).")
    
    num_instances, K, _ = keypoints.shape
    
    # Validate visibility tensor
    if visibility is not None:
        if not isinstance(visibility, torch.Tensor) or visibility.shape != (num_instances, K):
            raise ValueError("Visibility must be a tensor of shape (num_instances, K).")
    
    # Draw keypoints and connections
    for i in range(num_instances):
        for k in range(K):
            if visibility is None or visibility[i, k]:
                x, y = keypoints[i, k].tolist()
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors, outline=colors)
        
        if connectivity:
            for start, end in connectivity:
                if (visibility is None or (visibility[i, start] and visibility[i, end])):
                    x0, y0 = keypoints[i, start].tolist()
                    x1, y1 = keypoints[i, end].tolist()
                    draw.line((x0, y0, x1, y1), fill=colors, width=width)
    
    # Convert PIL Image back to tensor
    result_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1)
    
    return result_image

