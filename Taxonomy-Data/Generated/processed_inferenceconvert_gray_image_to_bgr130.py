import cv2
import numpy as np

def convert_gray_image_to_bgr(gray_image):
    """
    Convert a grayscale image to BGR format.

    Parameters:
    gray_image (numpy.ndarray): Input grayscale image.

    Returns:
    numpy.ndarray: BGR image.
    """
    # Check if the input image is a grayscale image
    if len(gray_image.shape) == 2 or (len(gray_image.shape) == 3 and gray_image.shape[2] == 1):
        # Convert the grayscale image to BGR
        bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        return bgr_image
    else:
        raise ValueError("Input image is not a valid grayscale image.")

