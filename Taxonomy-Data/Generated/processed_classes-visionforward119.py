import random
import numbers
from typing import Sequence
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch

class RandomPerspectiveTransform:
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        self.p = p

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def forward(self, img):
        if random.random() < self.p:
            width, height = self._get_image_size(img)
            startpoints, endpoints = self._get_params(width, height, self.distortion_scale)
            img = F.perspective(img, startpoints, endpoints, self.interpolation, self.fill)
        return img

    def _get_image_size(self, img):
        if isinstance(img, torch.Tensor):
            return img.shape[-1], img.shape[-2]
        elif isinstance(img, Image.Image):
            return img.size
        else:
            raise TypeError("Input should be a PIL Image or a Tensor")

    def _get_params(self, width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))

        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]

        return startpoints, endpoints

