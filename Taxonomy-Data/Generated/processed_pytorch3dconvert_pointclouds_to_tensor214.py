import torch

# Assuming Pointclouds is a class from a library like PyTorch3D
class Pointclouds:
    def __init__(self, points):
        self.points = points  # This is a placeholder for the actual implementation

    def to_padded_tensor(self):
        # Placeholder method to convert pointclouds to a padded tensor
        # In practice, this would convert the internal point cloud data to a tensor
        return torch.tensor(self.points)

    def num_points_per_cloud(self):
        # Placeholder method to get the number of points per cloud
        # In practice, this would return a list or tensor of the number of points in each cloud
        return [len(p) for p in self.points]

def convert_pointclouds_to_tensor(pcl):
    if isinstance(pcl, torch.Tensor):
        # If pcl is a torch.Tensor, return it and the number of points
        num_points = pcl.size(1)
        return pcl, num_points
    elif isinstance(pcl, Pointclouds):
        # If pcl is a Pointclouds object, convert to padded tensor and get num points
        padded_tensor = pcl.to_padded_tensor()
        num_points = pcl.num_points_per_cloud()
        return padded_tensor, num_points
    else:
        # Raise an error if the input is neither a torch.Tensor nor a Pointclouds object
        raise ValueError("Input must be a torch.Tensor or a Pointclouds object.")

