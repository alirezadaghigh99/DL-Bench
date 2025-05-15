import torch
import torch.nn.functional as F
from torch import nn, Tensor

class RCF:
    def __init__(self, in_channels: int = 4, features: int = 16, kernel_size: int = 3, bias: float = -1.0, seed: int | None = None, mode: str = 'gaussian', dataset: None = None):
        # Initialize weights and biases for two convolution layers
        torch.manual_seed(seed)
        self.weights1 = nn.Parameter(torch.randn(features, in_channels, kernel_size, kernel_size))
        self.biases1 = nn.Parameter(torch.full((features,), bias))
        self.weights2 = nn.Parameter(torch.randn(features, features, kernel_size, kernel_size))
        self.biases2 = nn.Parameter(torch.full((features,), bias))
        self.num_features = features * 2  # Since we concatenate two feature maps

    def forward(self, x: Tensor) -> Tensor:
        # First convolution + ReLU
        x1 = F.conv2d(x, self.weights1, self.biases1, padding=1)
        x1 = F.relu(x1)
        
        # Second convolution + ReLU
        x2 = F.conv2d(x1, self.weights2, self.biases2, padding=1)
        x2 = F.relu(x2)
        
        # Adaptive average pooling to a single value per feature map
        pooled1 = F.adaptive_avg_pool2d(x1, (1, 1))
        pooled2 = F.adaptive_avg_pool2d(x2, (1, 1))
        
        # Flatten the pooled outputs
        pooled1 = pooled1.view(pooled1.size(0), -1)
        pooled2 = pooled2.view(pooled2.size(0), -1)
        
        # Concatenate along the feature dimension
        output = torch.cat((pooled1, pooled2), dim=1)
        
        return output