import torch

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    # Convert inputs to tensors and ensure they are on the correct device
    distance = torch.tensor(distance, device=device, dtype=torch.float32).reshape(-1, 1)
    elevation = torch.tensor(elevation, device=device, dtype=torch.float32).reshape(-1, 1)
    azimuth = torch.tensor(azimuth, device=device, dtype=torch.float32).reshape(-1, 1)
    
    # Convert angles from degrees to radians if necessary
    if degrees:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)
    
    # Calculate the Cartesian coordinates
    x = distance * torch.cos(elevation) * torch.sin(azimuth)
    y = distance * torch.sin(elevation)
    z = distance * torch.cos(elevation) * torch.cos(azimuth)
    
    # Concatenate the results into a single tensor of shape (N, 3)
    camera_positions = torch.cat((x, y, z), dim=1)
    
    return camera_positions

