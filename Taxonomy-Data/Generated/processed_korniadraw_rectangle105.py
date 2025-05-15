import torch

def draw_rectangle(image, rectangle, color=None, fill=False):
    # Check the batch size
    B, C, H, W = image.shape
    assert rectangle.shape[0] == B, "Batch size of image and rectangle must match."
    assert rectangle.shape[2] == 4, "Each rectangle must be defined by 4 coordinates (x1, y1, x2, y2)."
    
    # Default color is white
    if color is None:
        color = torch.tensor([255] * C, dtype=image.dtype, device=image.device)
    
    # Handle different shapes of the color tensor
    if color.dim() == 1:
        color = color.view(1, 1, -1)  # Shape (1, 1, C)
    elif color.dim() == 2:
        color = color.view(B, -1, 1)  # Shape (B, N, 1)
    elif color.dim() == 3:
        pass  # Shape (B, N, C)
    else:
        raise ValueError("Color tensor has an invalid shape.")
    
    # Broadcast color to match the number of rectangles
    if color.shape[-1] == 1 and C == 3:
        color = color.expand(-1, -1, 3)  # Broadcast to 3 channels if needed
    
    # Draw rectangles
    for b in range(B):
        for n in range(rectangle.shape[1]):
            x1, y1, x2, y2 = rectangle[b, n]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            rect_color = color[b, n]
            
            if fill:
                image[b, :, y1:y2, x1:x2] = rect_color.view(C, 1, 1)
            else:
                # Draw borders
                image[b, :, y1, x1:x2] = rect_color.view(C, 1)
                image[b, :, y2-1, x1:x2] = rect_color.view(C, 1)
                image[b, :, y1:y2, x1] = rect_color.view(C, 1)
                image[b, :, y1:y2, x2-1] = rect_color.view(C, 1)
    
    return image

