import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super(FCN, self).__init__()
        
        # Define the layers of the FCN
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_filters, classes, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x