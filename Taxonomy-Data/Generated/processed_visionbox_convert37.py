import torch

def box_convert(boxes, in_fmt, out_fmt):
    if in_fmt == out_fmt:
        return boxes.clone()

    if in_fmt not in ['xyxy', 'xywh', 'cxcywh']:
        raise ValueError(f"Unsupported in_fmt: {in_fmt}")
    if out_fmt not in ['xyxy', 'xywh', 'cxcywh']:
        raise ValueError(f"Unsupported out_fmt: {out_fmt}")

    if in_fmt == 'xyxy':
        x1, y1, x2, y2 = boxes.unbind(-1)
        if out_fmt == 'xywh':
            return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)
        elif out_fmt == 'cxcywh':
            return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)

    elif in_fmt == 'xywh':
        x1, y1, w, h = boxes.unbind(-1)
        if out_fmt == 'xyxy':
            return torch.stack((x1, y1, x1 + w, y1 + h), dim=-1)
        elif out_fmt == 'cxcywh':
            return torch.stack((x1 + w / 2, y1 + h / 2, w, h), dim=-1)

    elif in_fmt == 'cxcywh':
        cx, cy, w, h = boxes.unbind(-1)
        if out_fmt == 'xyxy':
            return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
        elif out_fmt == 'xywh':
            return torch.stack((cx - w / 2, cy - h / 2, w, h), dim=-1)

    raise ValueError(f"Conversion from {in_fmt} to {out_fmt} is not supported.")

