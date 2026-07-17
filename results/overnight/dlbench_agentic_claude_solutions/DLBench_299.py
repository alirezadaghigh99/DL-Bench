from typing import Any, Dict, Optional, Tuple, Union
from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.constants import Resample
from kornia.core import Tensor, pad
from kornia.geometry import crop_by_transform_mat3d, get_perspective_transform3d

class RandomCrop3D(GeometricAugmentationBase3D):
    """Apply random crop on 3D volumes (5D tensor).

    Crops random sub-volumes on a given size.

    Args:
        p: probability of applying the transformation for the whole batch.
        size: Desired output size (out_d, out_h, out_w) of the crop.
            Must be Tuple[int, int, int], then out_d = size[0], out_h = size[1], out_w = size[2].
        padding: Optional padding on each border of the image.
            Default is None, i.e no padding. If a sequence of length 6 is provided, it is used to pad
            left, top, right, bottom, front, back borders respectively.
            If a sequence of length 3 is provided, it is used to pad left/right,
            top/bottom, front/back borders, respectively.
        pad_if_needed: It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, , out_d, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 3, 3, 3)
        >>> aug = RandomCrop3D((2, 2, 2), p=1.)
        >>> aug(inputs)
        tensor([[[[[-1.1258, -1.1524],
                   [-0.4339,  0.8487]],
        <BLANKLINE>
                  [[-1.2633,  0.3500],
                   [ 0.1665,  0.8744]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomCrop3D((24, 24, 24), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, size: Tuple[int, int, int], padding: Optional[Union[int, Tuple[int, int, int], Tuple[int, int, int, int, int, int]]]=None, pad_if_needed: Optional[bool]=False, fill: int=0, padding_mode: str='constant', resample: Union[str, int, Resample]=Resample.BILINEAR.name, same_on_batch: bool=False, align_corners: bool=True, p: float=1.0, keepdim: bool=False) -> None:
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self.flags = {'size': size, 'padding': padding, 'pad_if_needed': pad_if_needed, 'padding_mode': padding_mode, 'fill': fill, 'resample': Resample.get(resample), 'align_corners': align_corners}
        self._param_generator = rg.CropGenerator3D(size, None)

    def precrop_padding(self, input: Tensor, flags: Optional[Dict[str, Any]]=None) -> Tensor:
        flags = self.flags if flags is None else flags

        padding = [0, 0, 0, 0, 0, 0]  # left, right, top, bottom, front, back
        if flags['padding'] is not None:
            if isinstance(flags['padding'], int):
                padding = [flags['padding']] * 6
            elif isinstance(flags['padding'], (tuple, list)) and len(flags['padding']) == 3:
                padding = [
                    flags['padding'][0], flags['padding'][0],
                    flags['padding'][1], flags['padding'][1],
                    flags['padding'][2], flags['padding'][2],
                ]
            elif isinstance(flags['padding'], (tuple, list)) and len(flags['padding']) == 6:
                padding = [
                    flags['padding'][0], flags['padding'][2],
                    flags['padding'][1], flags['padding'][3],
                    flags['padding'][4], flags['padding'][5],
                ]
            else:
                raise RuntimeError(f"Expect `padding` to be a scalar, or length 3/6 list. Got {flags['padding']}.")

        if flags['pad_if_needed']:
            needed_padding = (
                flags['size'][0] - input.shape[-3],  # depth
                flags['size'][1] - input.shape[-2],  # height
                flags['size'][2] - input.shape[-1],  # width
            )
            if needed_padding[2] > 0:
                padding[0] = max(needed_padding[2], padding[0])
                padding[1] = max(needed_padding[2], padding[1])
            if needed_padding[1] > 0:
                padding[2] = max(needed_padding[1], padding[2])
                padding[3] = max(needed_padding[1], padding[3])
            if needed_padding[0] > 0:
                padding[4] = max(needed_padding[0], padding[4])
                padding[5] = max(needed_padding[0], padding[5])

        if any(padding):
            input = pad(input, padding, value=flags['fill'], mode=flags['padding_mode'])

        return input

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        transform: Tensor = get_perspective_transform3d(params['src'].to(input), params['dst'].to(input))
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor]=None) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the transform to be a Tensor. Gotcha {type(transform)}')
        return crop_by_transform_mat3d(input, transform, flags['size'], mode=flags['resample'].name.lower(), align_corners=flags['align_corners'])

    def forward(self, input: Tensor, params: Optional[Dict[str, Tensor]]=None, **kwargs: Any) -> Tensor:
        input = self.precrop_padding(input)
        return super().forward(input, params)
