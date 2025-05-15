from PIL import Image
import numpy as np
import torch

class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            # Convert torch Tensor to numpy array
            pic = pic.mul(255).byte().cpu().numpy()
            if pic.ndim == 3:
                # Handle single image
                pic = np.transpose(pic, (1, 2, 0))  # C x H x W -> H x W x C
            elif pic.ndim == 4:
                # Handle batch of images
                pic = np.transpose(pic, (0, 2, 3, 1))  # B x C x H x W -> B x H x W x C
        elif not isinstance(pic, np.ndarray):
            raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

        if isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                # Grayscale image
                mode = self.mode if self.mode is not None else 'L'
            elif pic.ndim == 3 and pic.shape[2] == 1:
                # Grayscale image with single channel
                mode = self.mode if self.mode is not None else 'L'
                pic = pic[:, :, 0]
            elif pic.ndim == 3 and pic.shape[2] == 3:
                # RGB image
                mode = self.mode if self.mode is not None else 'RGB'
            elif pic.ndim == 3 and pic.shape[2] == 4:
                # RGBA image
                mode = self.mode if self.mode is not None else 'RGBA'
            else:
                raise ValueError(f"Unsupported number of channels: {pic.shape[2]}")

            return Image.fromarray(pic, mode=mode)
        else:
            raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")