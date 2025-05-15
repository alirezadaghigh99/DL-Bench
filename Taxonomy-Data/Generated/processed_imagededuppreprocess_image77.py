from PIL import Image
import numpy as np

def preprocess_image(image, target_size=None, grayscale=False):
    # Input Validation
    if isinstance(image, np.ndarray):
        # Convert NumPy array to Pillow Image
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a NumPy array or a Pillow Image object.")
    
    # Resizing
    if target_size is not None:
        image = image.resize(target_size, Image.ANTIALIAS)
    
    # Grayscale Conversion
    if grayscale:
        image = image.convert('L')
    
    # Convert back to NumPy array
    processed_image = np.array(image)
    
    return processed_image