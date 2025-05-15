import cv2
import base64
import numpy as np

def numpy_array_to_base64_jpeg(image: np.ndarray) -> str:
    # Check if the input is a valid NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    # Encode the image as a JPEG
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Image encoding failed.")
    
    # Convert the encoded image to a byte array
    byte_data = encoded_image.tobytes()
    
    # Encode the byte data to a Base64 string
    base64_encoded = base64.b64encode(byte_data).decode('utf-8')
    
    return base64_encoded

