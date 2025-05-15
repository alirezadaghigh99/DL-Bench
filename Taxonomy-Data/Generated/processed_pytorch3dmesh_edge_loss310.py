import torch
from pytorch3d.structures import Meshes

def mesh_edge_loss(meshes: Meshes, target_length: float) -> torch.Tensor:
    # Check if the meshes object is empty
    if len(meshes) == 0:
        return torch.tensor(0.0, device=meshes.device)

    # Initialize the total loss
    total_loss = 0.0

    # Get the number of meshes in the batch
    num_meshes = len(meshes)

    # Iterate over each mesh in the batch
    for mesh in meshes:
        # Get the edges of the mesh
        edges = mesh.edges_packed()
        
        # Get the vertices of the mesh
        verts = mesh.verts_packed()
        
        # Calculate the edge lengths
        edge_lengths = torch.norm(verts[edges[:, 0]] - verts[edges[:, 1]], dim=1)
        
        # Calculate the number of edges in the current mesh
        num_edges = edge_lengths.shape[0]
        
        # Calculate the weight for each edge
        weight = 1.0 / num_edges
        
        # Calculate the squared difference between edge lengths and target length
        edge_loss = (edge_lengths - target_length) ** 2
        
        # Calculate the weighted loss for the current mesh
        weighted_loss = weight * edge_loss.sum()
        
        # Add the weighted loss to the total loss
        total_loss += weighted_loss

    # Calculate the average loss across all meshes
    average_loss = total_loss / num_meshes

    return average_loss

