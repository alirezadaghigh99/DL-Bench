import torch
import cv2
import numpy as np

def undistort_image(image, K, dist):
    """
    Compensates an image for lens distortion.

    Parameters:
    - image: torch.Tensor, shape (*, C, H, W)
    - K: torch.Tensor, shape (*, 3, 3)
    - dist: torch.Tensor, shape (*, n)

    Returns:
    - undistorted_image: torch.Tensor, shape (*, C, H, W)
    """
    # Ensure the input is a batch of images
    assert image.dim() >= 3, "Image tensor must have at least 3 dimensions"
    assert K.dim() == 3, "Intrinsic matrix K must have 3 dimensions"
    assert dist.dim() == 2, "Distortion coefficients must have 2 dimensions"

    # Get the shape of the input image
    batch_size, channels, height, width = image.shape

    # Prepare the output tensor
    undistorted_images = torch.empty_like(image)

    # Process each image in the batch
    for i in range(batch_size):
        # Convert the image to a NumPy array
        img_np = image[i].permute(1, 2, 0).cpu().numpy()

        # Convert the intrinsic matrix and distortion coefficients to NumPy arrays
        K_np = K[i].cpu().numpy()
        dist_np = dist[i].cpu().numpy()

        # Undistort the image using OpenCV
        undistorted_img_np = cv2.undistort(img_np, K_np, dist_np)

        # Convert the undistorted image back to a PyTorch tensor
        undistorted_img_tensor = torch.from_numpy(undistorted_img_np).permute(2, 0, 1)

        # Store the result in the output tensor
        undistorted_images[i] = undistorted_img_tensor

    return undistorted_images

