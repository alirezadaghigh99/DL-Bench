from PIL import Image, UnidentifiedImageError
import io

class EncodingError(Exception):
    """Custom exception for encoding errors."""
    pass

def bytes_to_pillow_image(payload: bytes) -> Image.Image:
    """
    Converts a bytes object to a Pillow Image object.

    Parameters:
    - payload: bytes - The bytes object containing image data.

    Returns:
    - Image.Image - A Pillow Image object.

    Raises:
    - EncodingError: If the bytes cannot be decoded into a valid image.
    """
    try:
        # Use BytesIO to create a file-like object from the bytes
        image_stream = io.BytesIO(payload)
        # Attempt to open the image using Pillow
        image = Image.open(image_stream)
        # Ensure the image is loaded (this can raise an error if the image is not valid)
        image.load()
        return image
    except (UnidentifiedImageError, IOError):
        # Raise an EncodingError if the image cannot be opened
        raise EncodingError("Could not encode bytes to PIL image.")