import torchvision.transforms as transforms
from typing import Optional, Union, Tuple

class DINOCollateFunction(MultiViewCollateFunction):
    def __init__(
        self,
        global_crop_size=224,
        global_crop_scale=(0.4, 1.0),
        local_crop_size=96,
        local_crop_scale=(0.05, 0.4),
        n_local_views=6,
        hf_prob=0.5,
        vf_prob=0,
        rr_prob=0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob=0.8,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.2,
        cj_hue=0.1,
        random_gray_scale=0.2,
        gaussian_blur=(1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob=0.2,
        normalize=imagenet_normalize,
    ):
        # Define global view transformations
        global_transforms = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale),
            transforms.RandomHorizontalFlip(p=hf_prob),
            transforms.RandomVerticalFlip(p=vf_prob),
            transforms.RandomRotation(degrees=rr_degrees) if rr_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue) if cj_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomGrayscale(p=random_gray_scale),
            transforms.GaussianBlur(kernel_size=kernel_size or int(global_crop_size * kernel_scale), sigma=sigmas) if gaussian_blur[0] > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomApply([transforms.Solarize(128)], p=solarization_prob),
            normalize,
        ])

        # Define local view transformations
        local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale),
            transforms.RandomHorizontalFlip(p=hf_prob),
            transforms.RandomVerticalFlip(p=vf_prob),
            transforms.RandomRotation(degrees=rr_degrees) if rr_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast, saturation=cj_sat, hue=cj_hue) if cj_prob > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomGrayscale(p=random_gray_scale),
            transforms.GaussianBlur(kernel_size=kernel_size or int(local_crop_size * kernel_scale), sigma=sigmas) if gaussian_blur[1] > 0 else transforms.Lambda(lambda x: x),
            transforms.RandomApply([transforms.Solarize(128)], p=solarization_prob),
            normalize,
        ])

        # Store transformations
        self.global_transforms = global_transforms
        self.local_transforms = local_transforms
        self.n_local_views = n_local_views

    def __call__(self, image):
        # Apply global transformations
        global_views = [self.global_transforms(image) for _ in range(2)]
        
        # Apply local transformations
        local_views = [self.local_transforms(image) for _ in range(self.n_local_views)]
        
        return global_views + local_views