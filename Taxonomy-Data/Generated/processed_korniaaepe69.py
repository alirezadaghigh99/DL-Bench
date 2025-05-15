import torch

def aepe(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate the Average Endpoint Error (AEPE) between two flow maps.

    Args:
        input (torch.Tensor): The input flow map with shape (*, 2).
        target (torch.Tensor): The target flow map with shape (*, 2).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        torch.Tensor: The computed AEPE as a scalar or tensor based on the reduction method.
    """
    # Ensure the input and target have the same shape
    if input.shape != target.shape:
        raise ValueError("Input and target must have the same shape.")

    # Calculate the squared differences
    diff = input - target
    squared_diff = diff ** 2

    # Sum the squared differences along the last dimension (2D vectors)
    sum_squared_diff = squared_diff.sum(dim=-1)

    # Calculate the square root of the sum of squared differences
    endpoint_error = torch.sqrt(sum_squared_diff)

    # Apply the specified reduction method
    if reduction == 'none':
        return endpoint_error
    elif reduction == 'mean':
        return endpoint_error.mean()
    elif reduction == 'sum':
        return endpoint_error.sum()
    else:
        raise ValueError("Reduction must be one of 'none', 'mean', or 'sum'.")

