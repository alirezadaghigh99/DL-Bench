from __future__ import annotations
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

def total_variation(img: Tensor, reduction: str='sum') -> Tensor:
    """Generate a Python function called total_variation that computes the Total Variation of an input image tensor. The function takes in an image tensor with shape (*, H, W) and an optional reduction parameter that specifies whether to return the sum or mean of the output. The function returns a tensor with shape (*). The Total Variation is calculated by taking the absolute differences of neighboring pixels in the image tensor along the height and width dimensions. The output is then either summed or averaged based on the reduction parameter. The function includes error checking for input types and reduction options."""
    KORNIA_CHECK_IS_TENSOR(img)
    KORNIA_CHECK_SHAPE(img, ["*", "H", "W"])

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-2, -1)
    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    if reduction == "mean":
        res1 = res1.mean(dim=reduce_axes)
        res2 = res2.mean(dim=reduce_axes)
    elif reduction == "sum":
        res1 = res1.sum(dim=reduce_axes)
        res2 = res2.sum(dim=reduce_axes)
    else:
        raise NotImplementedError("Invalid reduction option.")

    return res1 + res2

class TotalVariation(Module):
    """Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(*, H, W)`.
        - Output: :math:`(*,)`.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def forward(self, img: Tensor) -> Tensor:
        return total_variation(img)
