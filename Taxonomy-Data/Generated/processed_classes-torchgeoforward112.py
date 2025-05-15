import torch
from torch import nn, Tensor

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super().__init__()

        conv1 = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.Sequential(
            conv1,
            nn.LeakyReLU(inplace=True),
            conv2,
            nn.LeakyReLU(inplace=True),
            conv3,
            nn.LeakyReLU(inplace=True),
            conv4,
            nn.LeakyReLU(inplace=True),
            conv5,
            nn.LeakyReLU(inplace=True),
        )

        self.last = nn.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pass the input through the backbone
        x = self.backbone(x)
        # Pass the result through the final convolutional layer
        x = self.last(x)
        return x