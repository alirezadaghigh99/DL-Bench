import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from typing import List, Optional, Tuple

class RandomAffine(torch.nn.Module):
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
        center=None,
    ):
        super().__init__()
        self.degrees = self._setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center

    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        # Determine image dimensions
        img_size = F._get_image_size(img)

        # Ensure fill is a tuple with the same number of elements as image channels
        num_channels = F._get_image_num_channels(img)
        if isinstance(self.fill, (int, float)):
            fill = [self.fill] * num_channels
        else:
            fill = self.fill

        # Get parameters for affine transformation
        angle, translations, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, img_size
        )

        # Apply affine transformation
        return F.affine(
            img, angle=angle, translate=translations, scale=scale, shear=shear,
            interpolation=self.interpolation, fill=fill, center=self.center
        )

    def _setup_angle(self, degrees, name, req_sizes):
        # This is a placeholder for the actual implementation of _setup_angle
        # which should validate and set up the degrees parameter.
        if isinstance(degrees, (list, tuple)) and len(degrees) == 2:
            return degrees
        else:
            raise ValueError(f"{name} should be a sequence of length {req_sizes[0]}.")

