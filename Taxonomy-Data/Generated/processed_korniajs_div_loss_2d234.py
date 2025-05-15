import torch
import torch.nn.functional as F

def js_div_loss_2d(pred, target, reduction='mean'):
    """
    Calculate the Jensen-Shannon divergence loss between two heatmaps.

    Args:
        pred (torch.Tensor): Predicted tensor with shape (B, N, H, W).
        target (torch.Tensor): Target tensor with shape (B, N, H, W).
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # Ensure the inputs are probability distributions
    pred = F.softmax(pred, dim=-1)
    target = F.softmax(target, dim=-1)

    # Calculate the mean distribution
    m = 0.5 * (pred + target)

    # Calculate the KL divergence for each distribution
    kl_pred_m = F.kl_div(pred.log(), m, reduction='none')
    kl_target_m = F.kl_div(target.log(), m, reduction='none')

    # Calculate the Jensen-Shannon divergence
    js_div = 0.5 * (kl_pred_m + kl_target_m)

    # Apply the specified reduction
    if reduction == 'mean':
        return js_div.mean()
    elif reduction == 'sum':
        return js_div.sum()
    elif reduction == 'none':
        return js_div
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Choose from 'none', 'mean', or 'sum'.")

