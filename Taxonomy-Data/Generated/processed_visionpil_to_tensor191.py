from PIL import Image
import torch
import numpy as np

def pil_to_tensor(pic):
    # Check if the input is a PIL Image
    if not isinstance(pic, Image.Image):
        raise TypeError("Input must be a PIL Image")

    # Handle accimage separately if needed
    # Assuming accimage is a type of PIL Image, we can check its type
    # and handle it accordingly. If accimage is not installed, this part
    # can be omitted or adjusted based on your specific requirements.
    try:
        import accimage
        if isinstance(pic, accimage.Image):
            # Convert accimage to a numpy array and then to a tensor
            np_array = np.array(pic)
            return torch.tensor(np_array, dtype=torch.uint8)
    except ImportError:
        # accimage is not installed, continue with PIL Image processing
        pass

    # Convert PIL Image to a numpy array
    np_array = np.array(pic)

    # Convert numpy array to a tensor
    tensor = torch.tensor(np_array)

    # Rearrange dimensions from HWC to CHW
    if len(tensor.shape) == 3:  # Check if the image has 3 channels
        tensor = tensor.permute(2, 0, 1)

    return tensor

