import torch
from PIL import Image
import io

def encode_jpeg(images, quality=75):
    """
    Encodes an input tensor or list of tensors representing images into JPEG format.
    
    Parameters:
    - images: A tensor or list of tensors in CHW layout (Channel, Height, Width).
    - quality: An integer between 1 and 100 to control the output JPEG quality.
    
    Returns:
    - A tensor or list of tensors containing the raw bytes of the JPEG file(s).
    
    Raises:
    - ValueError: If the quality is not between 1 and 100 or if an empty list is passed as input.
    """
    if not (1 <= quality <= 100):
        raise ValueError("Quality must be between 1 and 100.")
    
    if isinstance(images, torch.Tensor):
        images = [images]
    
    if not images:
        raise ValueError("Input list of images is empty.")
    
    jpeg_bytes_list = []
    
    for img_tensor in images:
        if img_tensor.ndim != 3 or img_tensor.size(0) not in [1, 3]:
            raise ValueError("Each image tensor must be in CHW format with 1 or 3 channels.")
        
        # Convert CHW to HWC and scale to [0, 255]
        img = img_tensor.permute(1, 2, 0).clamp(0, 1) * 255
        img = img.byte().numpy()
        
        # Convert to PIL Image
        if img_tensor.size(0) == 1:
            img = img.squeeze(2)  # Remove channel dimension for grayscale
            pil_img = Image.fromarray(img, mode='L')
        else:
            pil_img = Image.fromarray(img, mode='RGB')
        
        # Encode to JPEG
        with io.BytesIO() as output:
            pil_img.save(output, format='JPEG', quality=quality)
            jpeg_bytes = output.getvalue()
            jpeg_bytes_list.append(torch.tensor(list(jpeg_bytes), dtype=torch.uint8))
    
    if len(jpeg_bytes_list) == 1:
        return jpeg_bytes_list[0]
    else:
        return jpeg_bytes_list

