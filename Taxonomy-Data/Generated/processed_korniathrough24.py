import torch

class ParametrizedLine:
    def __init__(self, point, direction):
        self.point = point  # A point on the line
        self.direction = direction  # The direction vector of the line

    @staticmethod
    def through(p0, p1):
        """
        Constructs a parametrized line between two points p0 and p1.

        Parameters:
        p0 (torch.Tensor): A tensor of shape (B, D) representing the starting points.
        p1 (torch.Tensor): A tensor of shape (B, D) representing the ending points.

        Returns:
        ParametrizedLine: An instance of ParametrizedLine representing the line.
        """
        # Calculate the direction vector from p0 to p1
        direction = p1 - p0
        
        # Normalize the direction vector
        direction_norm = torch.norm(direction, dim=1, keepdim=True)
        normalized_direction = direction / direction_norm
        
        # Return an instance of ParametrizedLine
        return ParametrizedLine(point=p0, direction=normalized_direction)

