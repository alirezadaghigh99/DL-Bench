import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class DiceLoss(nn.Module):
    def __init__(self, average: str = "micro", eps: float = 1e-8, weight: Optional[Tensor] = None) -> None:
        super(DiceLoss, self).__init__()
        self.average = average
        self.eps = eps
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Ensure predictions are probabilities
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode the target
        num_classes = pred.shape[1]
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        # Calculate Dice score
        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
        
        if self.average == 'micro':
            # Micro: Calculate the Dice loss across all classes
            dice_loss = 1.0 - torch.mean(dice_score)
        elif self.average == 'macro':
            # Macro: Calculate the Dice loss for each class separately and average
            if self.weight is not None:
                # Apply class weights if provided
                dice_loss = 1.0 - torch.sum(self.weight * torch.mean(dice_score, dim=0)) / torch.sum(self.weight)
            else:
                dice_loss = 1.0 - torch.mean(dice_score)
        else:
            raise ValueError("average must be either 'micro' or 'macro'")
        
        return dice_loss