import numpy as np
import cv2

class EncodingError(Exception):
    """Custom exception to be raised when image decoding fails."""
    pass

def bytes_to_opencv_image(payload, array_type=np.uint8):
    """
    Converts a bytes object into an OpenCV image represented as a numpy array.

    Parameters:
    - payload: bytes object containing the image data.
    - array_type: numpy data type for the array, default is np.uint8.

    Returns:
    - A numpy array representing the decoded OpenCV image.

    Raises:
    - EncodingError: If the image cannot be decoded.
    """
    # Convert the bytes object to a numpy array
    np_array = np.frombuffer(payload, dtype=array_type)
    
    # Decode the numpy array into an OpenCV image
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    
    # Check if the image was decoded successfully
    if image is None:
        raise EncodingError("Failed to decode image from bytes.")
    
    return image