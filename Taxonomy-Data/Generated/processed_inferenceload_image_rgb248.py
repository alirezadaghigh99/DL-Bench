import numpy as np
import cv2

def load_image(value, disable_preproc_auto_orient):
    """
    Mock function to simulate loading an image.
    This function should return a tuple (image_array, is_bgr).
    For demonstration, let's assume it always returns a BGR image.
    """
    # For demonstration purposes, let's create a dummy BGR image
    # In a real scenario, this function would load an image from a file or other source
    dummy_bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)  # A 100x100 black image
    is_bgr = True  # Assume the image is in BGR format
    return dummy_bgr_image, is_bgr

def load_image_rgb(value, disable_preproc_auto_orient=False):
    # Load the image using the load_image function
    image, is_bgr = load_image(value, disable_preproc_auto_orient)
    
    # Check if the image is in BGR format
    if is_bgr:
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # If the image is already in RGB format, no conversion is needed
        image_rgb = image
    
    return image_rgb

