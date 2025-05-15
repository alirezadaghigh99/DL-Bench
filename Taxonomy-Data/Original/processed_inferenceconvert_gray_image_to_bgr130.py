def convert_gray_image_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to BGR format.

    Args:
        image (np.ndarray): The grayscale image.

    Returns:
        np.ndarray: The converted BGR image.
    """

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image