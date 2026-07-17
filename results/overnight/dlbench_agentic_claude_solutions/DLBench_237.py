from kornia.core import Tensor, stack
from kornia.core.check import KORNIA_CHECK_IS_COLOR, KORNIA_CHECK_IS_TENSOR

def shift_rgb(image: Tensor, r_shift: Tensor, g_shift: Tensor, b_shift: Tensor) -> Tensor:
    """Generate a Python function named shift_rgb that shifts the RGB channels of an image. The function takes the following inputs: an image tensor of shape (N, C, H, W), where C is 3 (indicating an RGB image), and three tensors r_shift, g_shift, and b_shift of shape (N) that represent the shift values for the red, green, and blue channels, respectively.

The function should:

Verify that the image is a valid tensor and a color image.
Apply the shifts to each channel.
Ensure the output values are clamped between 0 and 1.
Return the modified image as a tensor of the same shape.
The function should handle errors by checking the tensor types and confirming that the input image is an RGB image."
example of input : image = tensor([[[[0.2000, 0.0000]],

         [[0.3000, 0.5000]],

         [[0.4000, 0.7000]]],


        [[[0.2000, 0.7000]],

         [[0.0000, 0.8000]],

         [[0.2000, 0.3000]]]]), r_shift = tensor([0.1000]), g_shift = tensor([0.3000]), b_shift = tensor([-0.3000])"""
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK_IS_COLOR(image)

    shifts = (r_shift, g_shift, b_shift)
    shifted = stack(
        [image[:, i, ...] + shifts[i].view(-1, 1, 1) for i in range(3)],
        dim=1,
    )

    return shifted.clamp(min=0.0, max=1.0)
