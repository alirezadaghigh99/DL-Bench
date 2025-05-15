import torch
from PIL import Image, ImageDraw, ImageFont
import warnings

def draw_bounding_boxes(image, boxes, labels=None, colors=None, fill=False, box_width=2, font=None, font_size=10):
    # Error handling for input types and shapes
    if not isinstance(image, torch.Tensor):
        raise TypeError("The image must be a PyTorch tensor.")
    
    if image.dtype not in [torch.uint8, torch.float32, torch.float64]:
        raise ValueError("The image dtype must be uint8 or float.")
    
    if image.ndim != 3 or image.shape[0] not in [1, 3]:
        raise ValueError("The image must have shape (C, H, W) with C being 1 or 3.")
    
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Bounding boxes must be a 2D tensor with shape (N, 4).")
    
    if labels is not None and len(labels) != len(boxes):
        warnings.warn("The number of labels does not match the number of boxes.")
    
    if len(boxes) == 0:
        warnings.warn("No bounding boxes provided.")
        return image

    # Convert image to PIL format
    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    
    if image.shape[0] == 1:  # Grayscale
        image_pil = Image.fromarray(image.squeeze(0).numpy(), mode='L')
    else:  # RGB
        image_pil = Image.fromarray(image.permute(1, 2, 0).numpy(), mode='RGB')
    
    draw = ImageDraw.Draw(image_pil, "RGBA" if fill else "RGB")

    # Default color
    default_color = (255, 0, 0, 128) if fill else (255, 0, 0)

    # Load font
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # Draw each bounding box
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.tolist()
        color = colors[i] if colors and i < len(colors) else default_color
        
        if fill:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, fill=color, width=box_width)
        else:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_width)
        
        if labels and i < len(labels):
            text_size = draw.textsize(labels[i], font=font)
            text_location = (xmin, ymin - text_size[1])
            draw.rectangle([text_location, (xmin + text_size[0], ymin)], fill=color)
            draw.text((xmin, ymin - text_size[1]), labels[i], fill=(255, 255, 255), font=font)

    # Convert back to tensor
    image_with_boxes = torch.from_numpy(np.array(image_pil))
    if image.shape[0] == 1:
        image_with_boxes = image_with_boxes.unsqueeze(0)
    else:
        image_with_boxes = image_with_boxes.permute(2, 0, 1)

    return image_with_boxes

