import torch
import math

def look_at_view_transform1(dist=1.0, elev=0.0, azim=0.0, degrees=True, eye=None, at=(0, 0, 0), up=(0, 1, 0), device="cpu"):
    # Convert inputs to tensors
    at = torch.tensor(at, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    
    if eye is None:
        # Convert angles to radians if they are in degrees
        if degrees:
            elev = math.radians(elev)
            azim = math.radians(azim)
        
        # Calculate the camera position in world coordinates
        eye_x = dist * math.cos(elev) * math.sin(azim)
        eye_y = dist * math.sin(elev)
        eye_z = dist * math.cos(elev) * math.cos(azim)
        eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float32, device=device)
    else:
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    
    # Calculate the forward vector from the camera to the object
    forward = at - eye
    forward = forward / torch.norm(forward)
    
    # Calculate the right vector
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    
    # Recalculate the up vector
    up = torch.cross(forward, right)
    
    # Create the rotation matrix
    R = torch.stack([right, up, forward], dim=0)
    
    # Create the translation matrix
    T = -R @ eye
    
    return R, T

