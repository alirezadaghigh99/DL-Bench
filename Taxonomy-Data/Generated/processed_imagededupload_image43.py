from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def load_image(image_file, target_size=None, grayscale=False, img_formats=None):
    """
    Load an image from a specified path and return an array version of the image.

    :param image_file: Path to the image file.
    :param target_size: Tuple (width, height) to resize the input image to (optional).
    :param grayscale: Boolean indicating whether to convert the image to grayscale (optional).
    :param img_formats: List of allowed image formats that can be loaded.
    :return: Processed image as a numpy array or None if an error occurs or format is not allowed.
    """
    try:
        # Open the image file
        with Image.open(image_file) as img:
            # Check if the image format is allowed
            if img_formats and img.format not in img_formats:
                logger.warning(f"Image format '{img.format}' is not allowed.")
                return None

            # Convert to grayscale if specified
            if grayscale:
                img = img.convert('L')  # 'L' mode is for grayscale

            # Resize the image if target_size is specified
            if target_size:
                img = img.resize(target_size, Image.ANTIALIAS)

            # Convert the image to a numpy array
            img_array = np.array(img)
            return img_array

    except Exception as e:
        logger.warning(f"An error occurred while loading the image: {e}")
        return None