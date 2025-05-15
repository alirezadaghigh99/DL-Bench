import torch

def _calculate_knots(lengths: torch.Tensor, lower: float, upper: float):
    # Calculate the cumulative sum of the lengths
    cumulative_lengths = torch.cumsum(lengths, dim=0)
    
    # Get the total length
    total_length = cumulative_lengths[-1]
    
    # Scale the cumulative lengths to fit between 0 and 1
    scaled_cumulative_lengths = cumulative_lengths / total_length
    
    # Scale and shift to fit between lower and upper
    knot_positions = lower + (upper - lower) * scaled_cumulative_lengths
    
    # Adjusted lengths are the differences between consecutive knot positions
    adjusted_lengths = torch.diff(torch.cat((torch.tensor([lower]), knot_positions)))
    
    return adjusted_lengths, knot_positions

