from pathlib import Path
import torch
from PIL import Image
import kornia as K

def _load_image_to_tensor(path_file: Path, device: torch.device) -> torch.Tensor:
    # Check if the file exists
    if not path_file.exists():
        raise FileNotFoundError(f"The file {path_file} does not exist.")
    
    # Open the image using PIL
    try:
        with Image.open(path_file) as img:
            # Convert the image to RGB (if not already in that format)
            img = img.convert('RGB')
            # Convert the image to a PyTorch tensor
            img_tensor = K.image_to_tensor(img, keepdim=False)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    # Move the tensor to the specified device
    img_tensor = img_tensor.to(device)
    
    return img_tensor

