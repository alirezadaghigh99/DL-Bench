import torch

def draw_line(image, p1, p2, color):
    # Check input dimensions
    assert image.ndim == 3, "Image must have 3 dimensions (C, H, W)"
    C, H, W = image.shape
    assert color.shape == (C,), "Color must have the same number of channels as the image"
    
    # Ensure p1 and p2 are tensors
    p1 = torch.tensor(p1, dtype=torch.int32)
    p2 = torch.tensor(p2, dtype=torch.int32)
    
    # Check if p1 and p2 are batched
    if p1.ndim == 1:
        p1 = p1.unsqueeze(0)  # Convert to (1, 2)
    if p2.ndim == 1:
        p2 = p2.unsqueeze(0)  # Convert to (1, 2)
    
    assert p1.shape == p2.shape, "p1 and p2 must have the same shape"
    assert p1.shape[1] == 2, "Points must have shape (2,) or (B, 2)"
    
    # Iterate over each pair of points
    for start, end in zip(p1, p2):
        x1, y1 = start
        x2, y2 = end
        
        # Check if points are within bounds
        if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
            raise ValueError("Points must be within the bounds of the image")
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < W and 0 <= y1 < H:
                image[:, y1, x1] = color  # Set the color at the current point
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    return image

