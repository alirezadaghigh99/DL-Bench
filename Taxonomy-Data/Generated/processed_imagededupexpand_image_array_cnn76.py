import numpy as np

def expand_image_array_cnn(image_arr):
    """
    Ensures the input image array is in the correct format for CNNs.
    
    Parameters:
    image_arr (np.ndarray): A NumPy array representing the input image. 
                            The array can be 2D (grayscale) or 3D (RGB).
    
    Returns:
    np.ndarray: A 3D NumPy array representing the image. If the input is a 2D array,
                it is converted to a 3D array by repeating the grayscale values across three channels.
    """
    if not isinstance(image_arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    if image_arr.ndim == 2:
        # Convert 2D grayscale image to 3D by repeating the grayscale values across three channels
        image_arr = np.stack((image_arr,) * 3, axis=-1)
    elif image_arr.ndim == 3:
        # Validate that the 3D array has three channels
        if image_arr.shape[-1] != 3:
            raise ValueError("3D input image must have three channels.")
    else:
        raise ValueError("Input image must be either 2D or 3D.")
    
    return image_arr

