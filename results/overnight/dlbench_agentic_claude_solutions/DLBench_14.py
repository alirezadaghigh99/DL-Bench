import torch

def mesh_edge_loss(meshes, target_length: float=0.0):
    """Create a Python function called mesh_edge_loss that computes the mesh edge length regularization loss averaged across all meshes in a batch. The function takes in two arguments: meshes (a Meshes object with a batch of meshes) and target_length (a float representing the resting value for the edge length). 

The function calculates the average loss across the batch, where each mesh contributes equally to the final loss, regardless of the number of edges per mesh. Each mesh is weighted with the inverse number of edges, so that meshes with fewer edges have a higher impact on the final loss.

If the input meshes object is empty, the function returns a tensor with a value of 0.0. Otherwise, the function computes the loss by determining the weight for each edge based on the number of edges in the corresponding mesh. The loss is calculated as the squared difference between the edge length and the target length, multiplied by the weights. 

Finally, the function returns the sum of the weighted losses divided by the total number of meshes in the batch."""
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()
    verts_packed = meshes.verts_packed()
    edges_packed_to_mesh_idx = meshes.edges_packed_to_mesh_idx()
    num_edges_per_mesh = meshes.num_edges_per_mesh()

    weights = num_edges_per_mesh.gather(0, edges_packed_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N
