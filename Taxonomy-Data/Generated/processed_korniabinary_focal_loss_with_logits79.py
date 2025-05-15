import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None, weight=None):
    """
    Compute the binary focal loss with logits.

    Args:
        pred (Tensor): Logits tensor with shape (N, C, *) where C = number of classes.
        target (Tensor): Labels tensor with the same shape as pred (N, C, *) where each value is between 0 and 1.
        alpha (float): Weighting factor alpha in [0, 1].
        gamma (float): Focusing parameter gamma >= 0.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        pos_weight (Tensor, optional): A weight of positive examples with shape (num_of_classes,).
        weight (Tensor, optional): Weights for classes with shape (num_of_classes,).

    Returns:
        Tensor: The computed loss.
    """
    # Compute the sigmoid of the predictions
    p = torch.sigmoid(pred)
    
    # Compute the focal loss components
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none', pos_weight=pos_weight)
    p_t = p * target + (1 - p) * (1 - target)  # p_t is the probability of the true class
    focal_loss = alpha * (1 - p_t) ** gamma * ce_loss

    # Apply class weights if provided
    if weight is not None:
        focal_loss = focal_loss * weight

    # Apply the specified reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

