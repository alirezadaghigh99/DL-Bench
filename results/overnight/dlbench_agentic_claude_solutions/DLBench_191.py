import torch
import torch.nn.functional as F
from ..utils import _log_api_usage_once

def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float=0.25, gamma: float=2, reduction: str='none') -> torch.Tensor:
    """Generate a Python function called sigmoid_focal_loss that calculates the focal loss used in RetinaNet for dense detection. The function takes in four parameters: inputs (a float tensor of arbitrary shape representing predictions), targets (a float tensor with the same shape as inputs representing binary classification labels), alpha (a float weighting factor to balance positive vs negative examples), gamma (a float exponent to balance easy vs hard examples), and reduction (a string specifying the reduction option for the output). The function returns a loss tensor with the specified reduction option applied. The function implements the focal loss formula and handles different reduction options such as 'none', 'mean', or 'sum'."""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss
