import torch
import torch.nn.functional as F

def kl_div_loss_2d(pred, target, reduction='mean'):
    """
    Calculate the Kullback-Leibler divergence loss between heatmaps.

    Args:
        pred: the input tensor with shape (B, N, H, W).
        target: the target tensor with shape (B, N, H, W).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        The KL divergence loss.
    """
    # Ensure the input tensors are in log space
    pred_log = torch.log(pred + 1e-10)  # Add a small value to avoid log(0)
    target_log = torch.log(target + 1e-10)

    # Calculate the KL divergence
    kl_div = F.kl_div(pred_log, target, reduction='none')

    # Apply the specified reduction
    if reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'none':
        return kl_div
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

