import torch
from enum import Enum

class BoundingBoxFormat(Enum):
    XYXY = 1
    XYWH = 2
    CXCYWH = 3

def _convert_to_xyxy(bounding_boxes, format):
    if format == BoundingBoxFormat.XYXY:
        return bounding_boxes
    elif format == BoundingBoxFormat.XYWH:
        x, y, w, h = bounding_boxes.unbind(-1)
        return torch.stack((x, y, x + w, y + h), dim=-1)
    elif format == BoundingBoxFormat.CXCYWH:
        cx, cy, w, h = bounding_boxes.unbind(-1)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
    else:
        raise ValueError("Unsupported bounding box format")

def _convert_from_xyxy(bounding_boxes, format):
    if format == BoundingBoxFormat.XYXY:
        return bounding_boxes
    elif format == BoundingBoxFormat.XYWH:
        x1, y1, x2, y2 = bounding_boxes.unbind(-1)
        return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)
    elif format == BoundingBoxFormat.CXCYWH:
        x1, y1, x2, y2 = bounding_boxes.unbind(-1)
        return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
    else:
        raise ValueError("Unsupported bounding box format")

def _clamp_bounding_boxes(bounding_boxes, format, canvas_size):
    # Convert to XYXY format
    xyxy_boxes = _convert_to_xyxy(bounding_boxes, format)
    
    # Clamp the coordinates
    height, width = canvas_size
    x1, y1, x2, y2 = xyxy_boxes.unbind(-1)
    x1 = x1.clamp(0, width)
    y1 = y1.clamp(0, height)
    x2 = x2.clamp(0, width)
    y2 = y2.clamp(0, height)
    clamped_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    
    # Convert back to the original format
    return _convert_from_xyxy(clamped_boxes, format)

