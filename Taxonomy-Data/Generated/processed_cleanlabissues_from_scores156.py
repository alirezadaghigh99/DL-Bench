import numpy as np
from typing import Optional, Union

def issues_from_scores(image_scores: np.ndarray, 
                       pixel_scores: Optional[np.ndarray] = None, 
                       threshold: float = None) -> Union[np.ndarray, np.ndarray]:
    # Validate inputs
    if image_scores is None:
        raise ValueError("image_scores cannot be None.")
    if threshold is None or not (0 <= threshold <= 1):
        raise ValueError("threshold must be a float between 0 and 1.")
    
    # If pixel_scores is provided, we need to return a boolean mask for pixel-level issues
    if pixel_scores is not None:
        if pixel_scores.shape[0] != image_scores.shape[0]:
            raise ValueError("The first dimension of pixel_scores must match the length of image_scores.")
        
        # Create a boolean mask where pixel scores are below the threshold
        pixel_issues_mask = pixel_scores < threshold
        return pixel_issues_mask
    
    # If pixel_scores is not provided, return indices of images with scores below the threshold
    else:
        image_issues_indices = np.where(image_scores < threshold)[0]
        return image_issues_indices

