from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
    """
    Convert a tensor or an ndarray to a PIL Image.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (str, optional): Mode for the PIL Image.

    Returns:
        PIL.Image.Image: Image converted to PIL Image.
    """
    if isinstance(pic, np.ndarray):
        # Handle numpy array
        if pic.ndim == 2:
            # Grayscale image
            return Image.fromarray(pic, mode=mode or 'L')
        elif pic.ndim == 3:
            # Color image
            if pic.shape[2] == 1:
                # Single channel image
                return Image.fromarray(pic.squeeze(2), mode=mode or 'L')
            elif pic.shape[2] == 3:
                # RGB image
                return Image.fromarray(pic, mode=mode or 'RGB')
            elif pic.shape[2] == 4:
                # RGBA image
                return Image.fromarray(pic, mode=mode or 'RGBA')
        else:
            raise ValueError(f"Unsupported numpy array shape: {pic.shape}")
    
    elif hasattr(pic, 'numpy'):
        # Handle PyTorch tensor
        pic = pic.numpy()
        return to_pil_image(pic, mode=mode)
    
    else:
        raise TypeError(f"Input type not supported: {type(pic)}")

